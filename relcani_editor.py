from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
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
    if clip.duration > 0:
        time = time % clip.duration

    if time <= clip.keyframes[0].time:
        return dict(clip.keyframes[0].bone_positions)
    if time >= clip.keyframes[-1].time:
        return dict(clip.keyframes[-1].bone_positions)

    for i in range(len(clip.keyframes) - 1):
        a = clip.keyframes[i]
        b = clip.keyframes[i + 1]
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
    return dict(clip.keyframes[-1].bone_positions)


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

    def set_time(self, value: float) -> None:
        self.current_time = min(max(value, 0.0), self.duration if self.duration > 0 else 0.0)

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


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RELCANi minimal standalone animation editor core")
    parser.add_argument("model", help="Path to proprietary model JSON (.relcani.json)")
    parser.add_argument("--animation", help="Animation clip to sample", default=None)
    parser.add_argument("--time", help="Timeline time to sample", type=float, default=0.0)
    parser.add_argument("--move", nargs=3, type=float, metavar=("DX", "DY", "DZ"), default=(0.0, 0.0, 0.0))
    return parser


def main() -> int:
    args = _build_cli().parse_args()
    model = ProprietaryModelLoader.load(args.model)
    obj = SceneObject(object_id="root", model=model)
    obj.move(args.move[0], args.move[1], args.move[2])

    if args.animation:
        clip = model.animations[args.animation]
        sampled = sample_animation(clip, args.time)
        print(f"Sampled animation '{args.animation}' at t={args.time:.3f}: {sampled}")

    viewport = ViewportRenderer()
    print(viewport.render_ascii(obj))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
