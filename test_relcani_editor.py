import json
import tempfile
import unittest
from pathlib import Path

from relcani_editor import ProprietaryModelLoader, SceneObject, Timeline, ViewportRenderer, sample_animation


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
        tmp = tempfile.NamedTemporaryFile("w", suffix=".relcani.json", delete=False)
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


if __name__ == "__main__":
    unittest.main()
