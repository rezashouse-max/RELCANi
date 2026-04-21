import json
import tempfile
import unittest
from pathlib import Path

from relcani_editor import (
    AnimationClip,
    AnimationKeyframe,
    Camera,
    EditorMode,
    EditorSession,
    LayoutMode,
    ProjectionType,
    ProprietaryModelLoader,
    SceneObject,
    Timeline,
    Viewport,
    ViewportLayout,
    ViewportRenderer,
    make_cube,
    make_sphere,
    sample_animation,
)


class RelcaniEditorTests(unittest.TestCase):
    def _write_model(self) -> str:
        model_data = {
            "name": "UnitTestModel",
            "textures": ["a.png"],
            "bones": [{"name": "root", "parent": None}],
            "vertices": [
                {
                    "position": [0.0, 0.0, 0.0],
                    "weights": [
                        {"bone": "root", "weight": 0.6},
                        {"bone": "b1", "weight": 0.2},
                        {"bone": "b2", "weight": 0.1},
                        {"bone": "b3", "weight": 0.05},
                        {"bone": "b4", "weight": 0.05},
                    ],
                }
            ],
            "animations": [
                {
                    "name": "walk",
                    "duration": 2.0,
                    "keyframes": [
                        {"time": 0.0, "bones": {"root": [0.0, 0.0, 0.0]}},
                        {"time": 2.0, "bones": {"root": [2.0, 0.0, 0.0]}},
                    ],
                }
            ],
        }
        tmp = tempfile.NamedTemporaryFile("w", suffix=".relcani.json", delete=False, encoding="utf-8")
        tmp.write(json.dumps(model_data))
        tmp.flush()
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return tmp.name

    def test_loader_normalizes_legacy_weights_to_top_4(self) -> None:
        path = self._write_model()
        model = ProprietaryModelLoader.load(path)
        weights = model.vertices[0].weights
        self.assertEqual(len(weights), 4)
        self.assertAlmostEqual(sum(w.weight for w in weights), 1.0, places=6)
        self.assertEqual([w.bone for w in weights], ["root", "b1", "b2", "b3"])

    def test_timeline_loops_when_playing(self) -> None:
        timeline = Timeline(duration=1.0)
        timeline.play()
        timeline.advance(1.4, loop=True)
        self.assertAlmostEqual(timeline.current_time, 0.4, places=6)

    def test_animation_sampling_interpolates(self) -> None:
        path = self._write_model()
        model = ProprietaryModelLoader.load(path)
        sampled = sample_animation(model.animations["walk"], 1.0)
        self.assertAlmostEqual(sampled["root"][0], 1.0, places=6)

    def test_scene_move_changes_viewport_output(self) -> None:
        path = self._write_model()
        model = ProprietaryModelLoader.load(path)
        obj = SceneObject(object_id="o1", model=model)
        viewport = ViewportRenderer(width=20, height=10)
        before = viewport.render_ascii(obj)
        obj.move(5.0, 0.0, 0.0)
        after = viewport.render_ascii(obj)
        self.assertNotEqual(before, after)

    def test_animation_sampling_supports_wrap_segment(self) -> None:
        path = self._write_model()
        model = ProprietaryModelLoader.load(path)
        base_clip = model.animations["walk"]
        clip = AnimationClip(
            name=base_clip.name,
            duration=2.0,
            keyframes=(
                AnimationKeyframe(time=0.5, bone_positions={"root": (0.5, 0.0, 0.0)}),
                AnimationKeyframe(time=1.5, bone_positions={"root": (1.5, 0.0, 0.0)}),
            ),
        )
        sampled = sample_animation(clip, 0.0)
        self.assertAlmostEqual(sampled["root"][0], 1.0, places=6)


class EditorLayoutTests(unittest.TestCase):
    def test_create_cube_and_sphere_appear_in_object_list(self) -> None:
        session = EditorSession()
        cube = session.create_cube()
        sphere = session.create_sphere()
        self.assertIn(cube.object_id, session.object_list())
        self.assertIn(sphere.object_id, session.object_list())
        self.assertNotEqual(cube.object_id, sphere.object_id)
        self.assertGreater(len(cube.model.vertices), 0)
        self.assertGreater(len(sphere.model.vertices), 0)

    def test_middle_click_toggles_quad_and_single_layout(self) -> None:
        layout = ViewportLayout.default()
        self.assertIs(layout.mode, LayoutMode.QUAD)
        layout.middle_click(ProjectionType.TOP)
        self.assertIs(layout.mode, LayoutMode.SINGLE)
        self.assertIs(layout.focused, ProjectionType.TOP)
        layout.middle_click(ProjectionType.TOP)
        self.assertIs(layout.mode, LayoutMode.QUAD)

    def test_quad_layout_renders_all_four_projection_labels(self) -> None:
        session = EditorSession()
        session.create_cube()
        rendered = session.layout.render_ascii(session.scene.objects.values())
        for proj in ("TOP", "LEFT", "RIGHT", "PERSPECTIVE"):
            self.assertIn(f"[{proj}]", rendered)

    def test_animation_mode_includes_timeline_in_render(self) -> None:
        session = EditorSession()
        session.create_cube()
        session.set_mode(EditorMode.ANIMATION)
        session.timeline = Timeline(duration=2.0)
        session.timeline.set_time(0.5)
        output = session.render()
        self.assertIn("Timeline", output)
        self.assertIn("Mode: animation", output)

    def test_modeling_mode_does_not_render_timeline(self) -> None:
        session = EditorSession()
        session.create_cube()
        self.assertNotIn("Timeline", session.render())

    def test_camera_orbit_clamps_pitch_and_dolly_has_minimum(self) -> None:
        cam = Camera(distance=2.0)
        cam.orbit(0.0, 100.0)
        self.assertLess(cam.pitch, 1.5708)
        cam.dolly(-1000.0)
        self.assertGreaterEqual(cam.distance, cam.min_distance)

    def test_top_viewport_projects_origin_to_center(self) -> None:
        vp = Viewport(ProjectionType.TOP, Camera(), width=11, height=11)
        cube = SceneObject(object_id="c", model=make_cube(size=0.0))
        rendered = vp.render_ascii([cube])
        # Strip the header line.
        body = rendered.split("\n", 1)[1].splitlines()
        self.assertEqual(body[5][5], "#")

    def test_sphere_primitive_has_expected_vertex_count(self) -> None:
        sphere = make_sphere(segments=8, rings=6)
        self.assertEqual(len(sphere.vertices), (6 + 1) * 8)


if __name__ == "__main__":
    unittest.main()
