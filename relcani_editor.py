from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class BoneWeight:
    bone: str
    weight: float


@dataclass(frozen=True)
class Vertex:
    position: Vec3
    weights: Tuple[BoneWeight, ...]


@dataclass(frozen=True)
class Bone:
    name: str
    parent: Optional[str]


@dataclass(frozen=True)
class AnimationKeyframe:
    time: float
    bone_positions: Dict[str, Vec3]


@dataclass(frozen=True)
class AnimationClip:
    name: str
    duration: float
    keyframes: Tuple[AnimationKeyframe, ...]


@dataclass(frozen=True)
class Model:
    name: str
    textures: Tuple[str, ...]
    bones: Dict[str, Bone]
    vertices: Tuple[Vertex, ...]
    animations: Dict[str, AnimationClip]


def _to_vec3(values: Iterable[float]) -> Vec3:
    x, y, z = list(values)
    return float(x), float(y), float(z)


def normalize_legacy_weights(raw_weights: Iterable[dict], max_influences: int = 4) -> Tuple[BoneWeight, ...]:
    filtered = [
        BoneWeight(str(item["bone"]), float(item["weight"]))
        for item in raw_weights
        if float(item.get("weight", 0.0)) > 0.0 and "bone" in item
    ]
    filtered.sort(key=lambda w: w.weight, reverse=True)
    trimmed = filtered[:max_influences]
    total = sum(w.weight for w in trimmed)
    if total <= 0:
        return tuple()
    return tuple(BoneWeight(w.bone, w.weight / total) for w in trimmed)


class ProprietaryModelLoader:
    @staticmethod
    def load(path: str | Path) -> Model:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        bones = {
            str(item["name"]): Bone(name=str(item["name"]), parent=item.get("parent"))
            for item in data.get("bones", [])
        }
        vertices: List[Vertex] = []
        for item in data.get("vertices", []):
            vertices.append(
                Vertex(
                    position=_to_vec3(item["position"]),
                    weights=normalize_legacy_weights(item.get("weights", [])),
                )
            )

        animations: Dict[str, AnimationClip] = {}
        for anim in data.get("animations", []):
            keyframes = []
            for frame in anim.get("keyframes", []):
                keyframes.append(
                    AnimationKeyframe(
                        time=float(frame["time"]),
                        bone_positions={bone_name: _to_vec3(vec) for bone_name, vec in frame.get("bones", {}).items()},
                    )
                )
            keyframes.sort(key=lambda f: f.time)
            clip = AnimationClip(
                name=str(anim["name"]),
                duration=float(anim["duration"]),
                keyframes=tuple(keyframes),
            )
            animations[clip.name] = clip

        return Model(
            name=str(data["name"]),
            textures=tuple(str(t) for t in data.get("textures", [])),
            bones=bones,
            vertices=tuple(vertices),
            animations=animations,
        )


def sample_animation(clip: AnimationClip, time: float) -> Dict[str, Vec3]:
    if not clip.keyframes:
        return {}
    keyframes = clip.keyframes
    first = keyframes[0]
    last = keyframes[-1]
    if len(keyframes) == 1:
        return dict(first.bone_positions)
    if clip.duration > 0:
        time = time % clip.duration

    if first.time <= time <= last.time:
        for i in range(len(keyframes) - 1):
            a = keyframes[i]
            b = keyframes[i + 1]
            if a.time <= time <= b.time:
                span = b.time - a.time
                if span <= 0:
                    return dict(b.bone_positions)
                t = (time - a.time) / span
                out: Dict[str, Vec3] = {}
                bones = set(a.bone_positions) | set(b.bone_positions)
                for bone in bones:
                    av = a.bone_positions.get(bone, (0.0, 0.0, 0.0))
                    bv = b.bone_positions.get(bone, av)
                    out[bone] = (
                        av[0] + (bv[0] - av[0]) * t,
                        av[1] + (bv[1] - av[1]) * t,
                        av[2] + (bv[2] - av[2]) * t,
                    )
                return out

    if clip.duration > 0:
        # Interpolate across the wrap segment from the last keyframe to the first keyframe.
        wrapped_time = time if time >= last.time else time + clip.duration
        wrap_end = first.time + clip.duration
        if last.time <= wrapped_time <= wrap_end and wrap_end > last.time:
            t = (wrapped_time - last.time) / (wrap_end - last.time)
            wrap_out: Dict[str, Vec3] = {}
            bones = set(last.bone_positions) | set(first.bone_positions)
            for bone in bones:
                av = last.bone_positions.get(bone, (0.0, 0.0, 0.0))
                bv = first.bone_positions.get(bone, av)
                wrap_out[bone] = (
                    av[0] + (bv[0] - av[0]) * t,
                    av[1] + (bv[1] - av[1]) * t,
                    av[2] + (bv[2] - av[2]) * t,
                )
            return wrap_out
    return dict(first.bone_positions if time < first.time else last.bone_positions)


@dataclass
class SceneObject:
    object_id: str
    model: Model
    position: Vec3 = (0.0, 0.0, 0.0)
    rotation: Vec3 = (0.0, 0.0, 0.0)
    scale: Vec3 = (1.0, 1.0, 1.0)

    def move(self, dx: float, dy: float, dz: float) -> None:
        self.position = (self.position[0] + dx, self.position[1] + dy, self.position[2] + dz)


@dataclass
class SceneObjectManager:
    objects: Dict[str, SceneObject] = field(default_factory=dict)

    def add(self, obj: SceneObject) -> None:
        self.objects[obj.object_id] = obj

    def get(self, object_id: str) -> SceneObject:
        if object_id not in self.objects:
            raise KeyError(f"Scene object '{object_id}' was not found")
        return self.objects[object_id]

    def move(self, object_id: str, dx: float, dy: float, dz: float) -> None:
        self.get(object_id).move(dx, dy, dz)


@dataclass
class Timeline:
    duration: float
    current_time: float = 0.0
    playing: bool = False

    def __post_init__(self) -> None:
        if self.duration < 0:
            self.duration = 0.0

    def set_time(self, value: float) -> None:
        self.current_time = min(max(value, 0.0), self.duration)

    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def advance(self, delta: float, loop: bool = True) -> None:
        if not self.playing:
            return
        new_time = self.current_time + delta
        if self.duration <= 0:
            self.current_time = 0.0
        elif loop:
            self.current_time = new_time % self.duration
        else:
            self.current_time = min(new_time, self.duration)


class ViewportRenderer:
    def __init__(self, width: int = 40, height: int = 20) -> None:
        self.width = width
        self.height = height

    def render_ascii(self, obj: SceneObject) -> str:
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        for vertex in obj.model.vertices:
            world_x = vertex.position[0] * obj.scale[0] + obj.position[0]
            world_z = vertex.position[2] * obj.scale[2] + obj.position[2]
            sx = int((world_x + 10.0) / 20.0 * (self.width - 1))
            sy = int((world_z + 10.0) / 20.0 * (self.height - 1))
            if 0 <= sx < self.width and 0 <= sy < self.height:
                grid[self.height - 1 - sy][sx] = "#"
        return "\n".join("".join(row) for row in grid)


# ---------------------------------------------------------------------------
# Primitive generators (used by the "File > Create" object-list workflow).
# ---------------------------------------------------------------------------


def _root_bone() -> Dict[str, Bone]:
    return {"root": Bone(name="root", parent=None)}


def _root_weighted_vertex(position: Vec3) -> Vertex:
    return Vertex(position=position, weights=(BoneWeight("root", 1.0),))


def make_cube(name: str = "Cube", size: float = 1.0) -> Model:
    h = size / 2.0
    positions: Tuple[Vec3, ...] = (
        (-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h),
        (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h),
    )
    return Model(
        name=name,
        textures=tuple(),
        bones=_root_bone(),
        vertices=tuple(_root_weighted_vertex(p) for p in positions),
        animations={},
    )


def make_sphere(name: str = "Sphere", radius: float = 1.0, segments: int = 8, rings: int = 6) -> Model:
    if segments < 3:
        segments = 3
    if rings < 2:
        rings = 2
    vertices: List[Vertex] = []
    for r in range(rings + 1):
        phi = math.pi * r / rings  # 0..pi
        y = radius * math.cos(phi)
        ring_radius = radius * math.sin(phi)
        for s in range(segments):
            theta = 2.0 * math.pi * s / segments
            x = ring_radius * math.cos(theta)
            z = ring_radius * math.sin(theta)
            vertices.append(_root_weighted_vertex((x, y, z)))
    return Model(
        name=name,
        textures=tuple(),
        bones=_root_bone(),
        vertices=tuple(vertices),
        animations={},
    )


def make_plane(name: str = "Plane", size: float = 5.0, divisions: int = 4) -> Model:
    if divisions < 1:
        divisions = 1
    h = size / 2.0
    step = size / divisions
    vertices: List[Vertex] = []
    for i in range(divisions + 1):
        for j in range(divisions + 1):
            x = -h + j * step
            z = -h + i * step
            vertices.append(_root_weighted_vertex((x, 0.0, z)))
    return Model(
        name=name,
        textures=tuple(),
        bones=_root_bone(),
        vertices=tuple(vertices),
        animations={},
    )


# ---------------------------------------------------------------------------
# Camera and multi-projection viewports.
# ---------------------------------------------------------------------------


class ProjectionType(str, Enum):
    TOP = "top"
    LEFT = "left"
    RIGHT = "right"
    PERSPECTIVE = "perspective"


@dataclass
class Camera:
    """Orbit camera around a target point.

    Supports the camera-control gestures the editor exposes per viewport:
    orbit (yaw/pitch), pan (move target), and dolly (zoom in/out).
    """

    target: Vec3 = (0.0, 0.0, 0.0)
    distance: float = 10.0
    yaw: float = 0.0      # radians, around Y
    pitch: float = 0.5    # radians, above XZ plane
    fov: float = math.radians(60.0)
    min_distance: float = 0.1

    def orbit(self, d_yaw: float, d_pitch: float) -> None:
        self.yaw += d_yaw
        max_pitch = math.pi / 2.0 - 1e-3
        self.pitch = max(-max_pitch, min(max_pitch, self.pitch + d_pitch))

    def pan(self, dx: float, dy: float, dz: float) -> None:
        self.target = (self.target[0] + dx, self.target[1] + dy, self.target[2] + dz)

    def dolly(self, delta: float) -> None:
        self.distance = max(self.min_distance, self.distance + delta)

    def position(self) -> Vec3:
        cp = math.cos(self.pitch)
        x = self.target[0] + self.distance * cp * math.sin(self.yaw)
        y = self.target[1] + self.distance * math.sin(self.pitch)
        z = self.target[2] + self.distance * cp * math.cos(self.yaw)
        return (x, y, z)


@dataclass
class Viewport:
    """A single viewport with a projection type and its own camera."""

    projection: ProjectionType
    camera: Camera = field(default_factory=Camera)
    width: int = 40
    height: int = 20

    def _project(self, world: Vec3) -> Optional[Tuple[int, int]]:
        wx, wy, wz = world
        tx, ty, tz = self.camera.target
        if self.projection is ProjectionType.TOP:
            # Looking down -Y: X right, Z up on screen.
            u = (wx - tx) / max(self.camera.distance, 1e-3)
            v = (wz - tz) / max(self.camera.distance, 1e-3)
        elif self.projection is ProjectionType.LEFT:
            # Looking along +X: Z right, Y up.
            u = (wz - tz) / max(self.camera.distance, 1e-3)
            v = (wy - ty) / max(self.camera.distance, 1e-3)
        elif self.projection is ProjectionType.RIGHT:
            # Looking along -X: -Z right, Y up.
            u = -(wz - tz) / max(self.camera.distance, 1e-3)
            v = (wy - ty) / max(self.camera.distance, 1e-3)
        else:  # PERSPECTIVE
            # Simple perspective: translate by -camera.position, project on Z.
            cx, cy, cz = self.camera.position()
            ex, ey, ez = wx - cx, wy - cy, wz - cz
            # Rotate by -yaw around Y so camera looks along its forward vector.
            cs, sn = math.cos(-self.camera.yaw), math.sin(-self.camera.yaw)
            rx = cs * ex + sn * ez
            rz = -sn * ex + cs * ez
            ry = ey
            # Pitch rotation around X.
            cp, sp = math.cos(-self.camera.pitch), math.sin(-self.camera.pitch)
            ry2 = cp * ry - sp * rz
            rz2 = sp * ry + cp * rz
            if rz2 >= -1e-3:
                return None  # behind/at camera
            f = 1.0 / math.tan(self.camera.fov / 2.0)
            u = f * rx / -rz2
            v = f * ry2 / -rz2
        # Map u,v in roughly [-1,1] to screen coords.
        sx = int((u + 1.0) * 0.5 * (self.width - 1))
        sy = int((1.0 - (v + 1.0) * 0.5) * (self.height - 1))
        if 0 <= sx < self.width and 0 <= sy < self.height:
            return sx, sy
        return None

    def render_ascii(self, objects: Iterable[SceneObject]) -> str:
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        for obj in objects:
            for vertex in obj.model.vertices:
                world = (
                    vertex.position[0] * obj.scale[0] + obj.position[0],
                    vertex.position[1] * obj.scale[1] + obj.position[1],
                    vertex.position[2] * obj.scale[2] + obj.position[2],
                )
                hit = self._project(world)
                if hit is not None:
                    sx, sy = hit
                    grid[sy][sx] = "#"
        label = self.projection.value.upper()
        header = f"[{label}]".ljust(self.width)
        return header + "\n" + "\n".join("".join(row) for row in grid)


class LayoutMode(str, Enum):
    QUAD = "quad"
    SINGLE = "single"


@dataclass
class ViewportLayout:
    """Quad layout (top/left/right/perspective) with single-port toggle.

    Middle-clicking a viewport switches to single-port view of just that port;
    middle-clicking again restores the quad layout.
    """

    viewports: Dict[ProjectionType, Viewport] = field(default_factory=dict)
    mode: LayoutMode = LayoutMode.QUAD
    focused: ProjectionType = ProjectionType.PERSPECTIVE

    @classmethod
    def default(cls, width: int = 40, height: int = 20) -> "ViewportLayout":
        # Use half size for quad cells so the composed view fits typical terminals.
        cell_w = max(10, width // 2)
        cell_h = max(6, height // 2)
        layout = cls(
            viewports={
                ProjectionType.TOP: Viewport(ProjectionType.TOP, Camera(pitch=math.pi / 2 - 0.01), cell_w, cell_h),
                ProjectionType.LEFT: Viewport(ProjectionType.LEFT, Camera(yaw=math.pi / 2, pitch=0.0), cell_w, cell_h),
                ProjectionType.RIGHT: Viewport(ProjectionType.RIGHT, Camera(yaw=-math.pi / 2, pitch=0.0), cell_w, cell_h),
                ProjectionType.PERSPECTIVE: Viewport(ProjectionType.PERSPECTIVE, Camera(), cell_w, cell_h),
            },
            focused=ProjectionType.PERSPECTIVE,
        )
        return layout

    def middle_click(self, projection: ProjectionType) -> LayoutMode:
        """Toggle between quad and single-port view focused on `projection`."""
        if self.mode is LayoutMode.QUAD:
            self.mode = LayoutMode.SINGLE
            self.focused = projection
        else:
            self.mode = LayoutMode.QUAD
        return self.mode

    def render_ascii(self, objects: Iterable[SceneObject]) -> str:
        objs = list(objects)
        if self.mode is LayoutMode.SINGLE:
            return self.viewports[self.focused].render_ascii(objs)
        # Quad: top-left=TOP, top-right=PERSPECTIVE, bottom-left=LEFT, bottom-right=RIGHT.
        order = (
            (ProjectionType.TOP, ProjectionType.PERSPECTIVE),
            (ProjectionType.LEFT, ProjectionType.RIGHT),
        )
        rows: List[str] = []
        for left_proj, right_proj in order:
            left = self.viewports[left_proj].render_ascii(objs).splitlines()
            right = self.viewports[right_proj].render_ascii(objs).splitlines()
            line_count = max(len(left), len(right))
            for i in range(line_count):
                l = left[i] if i < len(left) else ""
                r = right[i] if i < len(right) else ""
                rows.append(f"{l} | {r}")
            rows.append("-" * (len(rows[-1]) if rows else 10))
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Editor mode (modeling vs animation) and high-level editor session.
# ---------------------------------------------------------------------------


class EditorMode(str, Enum):
    MODELING = "modeling"
    ANIMATION = "animation"


@dataclass
class EditorSession:
    """High-level editor state: object list, viewport layout, mode, timeline.

    Mirrors the requested top-level layout: object list + 4 viewports, and in
    animation mode a timeline area is exposed at the bottom for keyframing.
    """

    scene: SceneObjectManager = field(default_factory=SceneObjectManager)
    layout: ViewportLayout = field(default_factory=ViewportLayout.default)
    mode: EditorMode = EditorMode.MODELING
    timeline: Timeline = field(default_factory=lambda: Timeline(duration=0.0))
    _next_id: int = 1

    def _allocate_id(self, base: str) -> str:
        object_id = f"{base}_{self._next_id}"
        self._next_id += 1
        return object_id

    def create_cube(self, name: Optional[str] = None) -> SceneObject:
        object_id = self._allocate_id("cube")
        obj = SceneObject(object_id=object_id, model=make_cube(name or object_id))
        self.scene.add(obj)
        return obj

    def create_sphere(self, name: Optional[str] = None) -> SceneObject:
        object_id = self._allocate_id("sphere")
        obj = SceneObject(object_id=object_id, model=make_sphere(name or object_id))
        self.scene.add(obj)
        return obj

    def load_model(self, path: str | Path, object_id: Optional[str] = None) -> SceneObject:
        model = ProprietaryModelLoader.load(path)
        oid = object_id or self._allocate_id(model.name.lower() or "model")
        obj = SceneObject(object_id=oid, model=model)
        self.scene.add(obj)
        # If the loaded model has animations, configure the timeline to the
        # longest clip so animation mode can scrub it immediately.
        if model.animations:
            longest = max(c.duration for c in model.animations.values())
            self.timeline = Timeline(duration=longest)
        return obj

    def set_mode(self, mode: EditorMode) -> None:
        self.mode = mode

    def object_list(self) -> List[str]:
        return list(self.scene.objects.keys())

    def render(self) -> str:
        objects = self.scene.objects.values()
        header = f"Mode: {self.mode.value} | Layout: {self.layout.mode.value} | Objects: {len(self.scene.objects)}"
        viewport_view = self.layout.render_ascii(objects)
        sections = [header, "Object list: " + (", ".join(self.object_list()) or "<empty>"), viewport_view]
        if self.mode is EditorMode.ANIMATION:
            sections.append(self._render_timeline())
        return "\n".join(sections)

    def _render_timeline(self, width: int = 40) -> str:
        duration = self.timeline.duration
        bar_width = max(10, width)
        if duration <= 0:
            bar = "[" + "-" * bar_width + "]"
            return f"Timeline (no clip)\n{bar}"
        ratio = self.timeline.current_time / duration if duration > 0 else 0.0
        cursor = min(bar_width - 1, max(0, int(ratio * (bar_width - 1))))
        bar_chars = ["-"] * bar_width
        bar_chars[cursor] = "|"
        bar = "[" + "".join(bar_chars) + "]"
        return (
            f"Timeline t={self.timeline.current_time:.3f}/{duration:.3f}"
            f" {'play' if self.timeline.playing else 'pause'}\n{bar}"
        )


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RELCANi minimal standalone animation editor core")
    parser.add_argument("model", nargs="?", help="Path to proprietary model JSON (.relcani.json)")
    parser.add_argument("--animation", help="Animation clip to sample", default=None)
    parser.add_argument("--time", help="Timeline time to sample", type=float, default=0.0)
    parser.add_argument("--move", nargs=3, type=float, metavar=("DX", "DY", "DZ"), default=(0.0, 0.0, 0.0))
    parser.add_argument(
        "--create",
        action="append",
        choices=("cube", "sphere"),
        default=[],
        help="Create a primitive in the editor object list (repeatable). Mirrors File > Create.",
    )
    parser.add_argument(
        "--mode",
        choices=(EditorMode.MODELING.value, EditorMode.ANIMATION.value),
        default=EditorMode.MODELING.value,
        help="Editor mode (mirrors the top-right menu strip selector).",
    )
    parser.add_argument(
        "--single-port",
        choices=tuple(p.value for p in ProjectionType),
        default=None,
        help="Start in single-port view of the given projection (otherwise quad layout).",
    )
    return parser


def main() -> int:
    args = _build_cli().parse_args()

    session = EditorSession()
    session.set_mode(EditorMode(args.mode))

    if args.model:
        obj = session.load_model(args.model)
        obj.move(args.move[0], args.move[1], args.move[2])

        if args.animation:
            model = obj.model
            if args.animation not in model.animations:
                available = ", ".join(sorted(model.animations)) or "<none>"
                raise SystemExit(
                    f"Animation '{args.animation}' was not found in model '{model.name}'. "
                    f"Available animations: {available}"
                )
            clip = model.animations[args.animation]
            sampled = sample_animation(clip, args.time)
            session.timeline.set_time(args.time)
            print(f"Sampled animation '{args.animation}' at t={args.time:.3f}: {sampled}")

    for primitive in args.create:
        if primitive == "cube":
            session.create_cube()
        else:
            session.create_sphere()

    if args.single_port is not None:
        session.layout.middle_click(ProjectionType(args.single_port))

    print(session.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
