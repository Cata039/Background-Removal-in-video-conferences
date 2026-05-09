# Background removal in video conferences — submission notes

This project builds on **[Robust Video Matting (RVM)](https://github.com/PeterL1n/RobustVideoMatting)** (GPL-3.0). Upstream source is preserved; this fork adds:

- `interface_app.py` — Gradio web UI (camera, alpha matte, composited background, experiment checkpoints)
- `finetune.py` — fine-tuning with resume, CSV logs, saved epochs
- `webcam_demo.py` — OpenCV demo
- `requirements_interface.txt` — UI dependencies
- `charts/` — comparison plots (where generated)
- `training_log*.csv` — per-experiment loss curves
- `ui_assets/backgrounds/` — optional background images for the UI

## What is **not** in GitHub (too large)

| Item | Why excluded | How to obtain |
|------|----------------|---------------|
| `*.pth` model weights | Each ~15 MB; dozens of checkpoints ≈ hundreds of MB | Download official **`rvm_mobilenetv3.pth`** from the [RVM releases](https://github.com/PeterL1n/RobustVideoMatting/releases/tag/v1.0.0). Place it in the project root. 
| `data/` (clips + mattes) | Hundreds of MB | - |

## Quick run (after clone)

```bash
pip install -r requirements_inference.txt
pip install -r requirements_interface.txt
# Download rvm_mobilenetv3.pth into this folder (see table above)
python interface_app.py
```

Open the URL printed in the terminal (Gradio).

## Remote layout

- `upstream` — original [PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)
- `origin` — your course submission repository
