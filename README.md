# RELCANi

Minimal standalone animation-editor core for:
- Loading a proprietary model format (`.relcani.json`) with textures, skeleton, and animations
- Managing scene objects and moving them in 3D space
- Scrubbing/playing animation with a timeline
- Rendering a simple ASCII 3D viewport preview

## Quick start

Create a model file (example):

```json
{
  "name": "Demo",
  "textures": ["body_diffuse.png"],
  "bones": [{"name": "root", "parent": null}],
  "vertices": [
    {
      "position": [0.0, 0.0, 0.0],
      "weights": [{"bone": "root", "weight": 1.0}]
    },
    {
      "position": [1.0, 0.0, 0.0],
      "weights": [{"bone": "root", "weight": 1.0}]
    }
  ],
  "animations": [
    {
      "name": "idle",
      "duration": 1.0,
      "keyframes": [
        {"time": 0.0, "bones": {"root": [0.0, 0.0, 0.0]}},
        {"time": 1.0, "bones": {"root": [0.0, 0.0, 0.0]}}
      ]
    }
  ]
}
```

Run:

```bash
python /home/runner/work/RELCANi/RELCANi/relcani_editor.py /path/to/model.relcani.json --animation idle --time 0.5
```
