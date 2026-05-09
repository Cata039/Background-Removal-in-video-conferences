import time
from pathlib import Path
import re
import subprocess
import os
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch

from model import MattingNetwork


ROOT_DIR = Path(__file__).resolve().parent
BACKGROUND_DIR = ROOT_DIR / "ui_assets" / "backgrounds"
BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
MODEL_CACHE: Dict[str, MattingNetwork] = {}
ACTIVE_CAPTURE: Optional[cv2.VideoCapture] = None
ACTIVE_CAMERA_INDEX: Optional[int] = None
EXPERIMENT_DEFS = {
    "Pretrained": {
        "label": "Pretrained",
        "patterns": [],
        "include_pretrained": True,
        "description": (
            "**Pretrained (no finetune)**\n\n"
            "- **What changes**: nothing (official pretrained weights).\n"
            "- **Why**: reference baseline to compare all fine-tuning experiments against."
        ),
    },
    "Experiment 1": {
        "label": "Experiment 1",
        "patterns": [
            "rvm_epoch*.pth",
            "finetuned_rvm_epoch_*.pth",
            "rvm_finetuned*.pth",
        ],
        "include_pretrained": False,
        "description": (
            "**Experiment 1 (baseline): fine-tune minimal**\n\n"
            "- **What changes**: only the **last 2 decoder layers** are trained (very few parameters).\n"
            "- **What stays frozen**: backbone (feature extractor) + most of the network.\n"
            "- **Why**: fastest to run and a good baseline; shows how far you can get with minimal adaptation.\n"
            "- **Expectation**: stable but limited improvement, especially on fine details."
        ),
    },
    "Experiment 2a": {
        "label": "Experiment 2a",
        "patterns": [
            # Second approach, LR=1e-4 run (default attempt2).
            "attempt2_epoch*.pth",
            "attempt2_finetuned*.pth",
        ],
        "include_pretrained": False,
        "description": (
            "**Experiment 2a: attempt2, LR=1e-4**\n\n"
            "- **Trainable**: **decoder + project_mat** (matting head).\n"
            "- **Frozen**: **backbone**.\n"
            "- **Why**: stronger adaptation than Exp 1, while keeping pretrained features stable.\n"
            "- **Difference vs 2b**: only **learning rate** (2a uses higher LR → faster, potentially less stable)."
        ),
    },
    "Experiment 2b": {
        "label": "Experiment 2b",
        "patterns": [
            # Same as attempt2 but with lower learning rate.
            "attempt2_lr1e-5_epoch*.pth",
            "attempt2_lr1e-5_finetuned*.pth",
        ],
        "include_pretrained": False,
        "description": (
            "**Experiment 2b: attempt2, LR=1e-5**\n\n"
            "- **Trainable**: **decoder + project_mat**.\n"
            "- **Frozen**: **backbone**.\n"
            "- **Why**: test LR sensitivity (lower LR → more gradual updates).\n"
            "- **Difference vs 2a**: only **learning rate** (2b uses lower LR → slower, often more stable)."
        ),
    },
    "Experiment 3a": {
        "label": "Experiment 3a",
        "patterns": [
            "exp1_bs2_epoch*.pth",
            "exp1_bs2_finetuned*.pth",
        ],
        "include_pretrained": False,
        "description": (
            "**Experiment 3a: batch size = 2 (Exp 1 setup)**\n\n"
            "- **Goal**: study batch-size impact on convergence/stability.\n"
            "- **Difference vs 3b/3c**: only **batch size** (2 vs 4 vs 8).\n"
            "- **Expectation**: noisier gradients; sometimes less stable but can generalize well."
        ),
    },
    "Experiment 3b": {
        "label": "Experiment 3b",
        "patterns": [
            # Your original Exp 1 checkpoints (batch size 4).
            "rvm_epoch*.pth",
            "finetuned_rvm_epoch_*.pth",
            "rvm_finetuned*.pth",
        ],
        "include_pretrained": False,
        "description": (
            "**Experiment 3b: batch size = 4 (Exp 1 setup)**\n\n"
            "- **This is your existing Exp 1 run**.\n"
            "- **Difference vs 3a/3c**: only **batch size** (4 as baseline).\n"
            "- **Expectation**: good balance between stability and speed."
        ),
    },
    "Experiment 3c": {
        "label": "Experiment 3c",
        "patterns": [
            "exp1_bs8_epoch*.pth",
            "exp1_bs8_finetuned*.pth",
        ],
        "include_pretrained": False,
        "description": (
            "**Experiment 3c: batch size = 8 (Exp 1 setup)**\n\n"
            "- **Goal**: compare against batch 2/4.\n"
            "- **Difference vs 3a/3b**: only **batch size** (larger batch → smoother gradients).\n"
            "- **Expectation**: more stable loss curve; may be limited by memory on MPS."
        ),
    },
    "All": {
        "label": "All",
        "patterns": [],
        "include_pretrained": True,
        "description": (
            "**All checkpoints**\n\n"
            "- Shows every checkpoint found in the project.\n"
            "- Useful for browsing; for clean comparisons use Pretrained vs Exp 1 vs 2a/2b vs 3a/3b/3c."
        ),
    },
}


def list_checkpoints_by_experiment() -> Dict[str, List[str]]:
    names_by_key: Dict[str, set] = {key: set() for key in EXPERIMENT_DEFS.keys()}

    # Collect experiment-specific.
    for key, exp in EXPERIMENT_DEFS.items():
        for pattern in exp["patterns"]:
            names_by_key[key].update(path.name for path in ROOT_DIR.glob(pattern))

    # Optionally include pretrained weight in specific experiments.
    pretrained = ROOT_DIR / "rvm_mobilenetv3.pth"
    if pretrained.exists():
        for key, exp in EXPERIMENT_DEFS.items():
            if exp.get("include_pretrained"):
                names_by_key[key].add(pretrained.name)

    # "All" is union of everything we found.
    union_all = set()
    for key in EXPERIMENT_DEFS.keys():
        if key != "All":
            union_all.update(names_by_key[key])
    names_by_key["All"] = union_all if union_all else names_by_key["All"]

    return {key: sorted(list(names), key=epoch_sort_key) for key, names in names_by_key.items()}


def checkpoints_for_experiment(experiment_label: str) -> Tuple[gr.Dropdown, dict, str]:
    grouped = list_checkpoints_by_experiment()
    choices = grouped.get(experiment_label, [])
    value = choices[0] if choices else None
    description = EXPERIMENT_DEFS.get(experiment_label, {}).get("description", "")
    return gr.Dropdown(choices=choices, value=value), {"checkpoint": None, "rec": [None] * 4}, description


def list_backgrounds() -> List[str]:
    bg_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        bg_files.extend(BACKGROUND_DIR.glob(ext))
    return sorted([path.name for path in bg_files])


def epoch_sort_key(name: str) -> Tuple[int, str]:
    if name == "rvm_mobilenetv3.pth":
        return (0, name)

    # Prefer parsing explicit epoch patterns to avoid mixing digits from prefixes like "attempt2".
    # Examples:
    # - rvm_epoch40.pth -> 40
    # - attempt2_epoch40.pth -> 40 (NOT 240)
    # - finetuned_rvm_epoch_12.pth -> 12
    match = re.search(r"(?:^|_)epoch_?(\d+)\.pth$", name)
    if match:
        epoch = int(match.group(1))
    else:
        epoch = 99999
    return (epoch, name)


def load_model(checkpoint_name: str) -> MattingNetwork:
    if checkpoint_name in MODEL_CACHE:
        return MODEL_CACHE[checkpoint_name]

    checkpoint_path = ROOT_DIR / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")

    model = MattingNetwork("mobilenetv3").eval().to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    MODEL_CACHE[checkpoint_name] = model
    return model


def open_camera(camera_index: int) -> str:
    global ACTIVE_CAPTURE, ACTIVE_CAMERA_INDEX

    if ACTIVE_CAPTURE is not None and ACTIVE_CAMERA_INDEX == camera_index and ACTIVE_CAPTURE.isOpened():
        return f"Camera {camera_index} is already active."

    if ACTIVE_CAPTURE is not None:
        ACTIVE_CAPTURE.release()

    # On macOS, explicitly prefer AVFoundation backend to keep indices consistent
    # with ffmpeg's avfoundation device list when available.
    backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else 0
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        ACTIVE_CAPTURE = None
        ACTIVE_CAMERA_INDEX = None
        return f"Cannot open camera index {camera_index}."

    ACTIVE_CAPTURE = cap
    ACTIVE_CAMERA_INDEX = camera_index
    return f"Using camera index {camera_index}."


def list_avfoundation_video_devices() -> List[Tuple[int, str]]:
    """
    Parse ffmpeg avfoundation device listing on macOS.
    Returns [(index, name), ...] for video devices.
    """
    try:
        proc = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []

    text = f"{proc.stdout}\n{proc.stderr}"
    devices: List[Tuple[int, str]] = []
    in_video = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "AVFoundation video devices" in line:
            in_video = True
            continue
        if "AVFoundation audio devices" in line:
            in_video = False
            continue
        if not in_video:
            continue

        match = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if match:
            devices.append((int(match.group(1)), match.group(2)))

    return devices


def get_preferred_mac_camera_index() -> Optional[int]:
    devices = list_avfoundation_video_devices()
    if not devices:
        return None

    # Prefer built-in webcam names and avoid Continuity/iPhone.
    preferred_keywords = ["facetime", "built-in", "integrated", "webcam", "macbook"]
    banned_keywords = [
        "continuity",
        "iphone",
        "ipad",
        "ios",
        "camera de continuitate",
        "cameră de continuitate",
        "continuity camera",
    ]

    for idx, name in devices:
        lowered = name.lower()
        if any(bad in lowered for bad in banned_keywords):
            continue
        if any(good in lowered for good in preferred_keywords):
            return idx

    # Fallback: first non-iPhone-looking device.
    for idx, name in devices:
        lowered = name.lower()
        if not any(bad in lowered for bad in banned_keywords):
            return idx

    return None


def save_uploaded_background(uploaded_image: np.ndarray) -> Tuple[gr.Dropdown, str]:
    if uploaded_image is None:
        return gr.Dropdown(choices=list_backgrounds()), "Upload an image first."

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = BACKGROUND_DIR / f"uploaded_{ts}.png"
    bgr = cv2.cvtColor(uploaded_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr)

    bg_choices = list_backgrounds()
    return gr.Dropdown(choices=bg_choices, value=out_path.name), f"Saved: {out_path.name}"


def render_frame(
    frame: np.ndarray,
    checkpoint_name: str,
    background_name: Optional[str],
    rec_state: Optional[dict],
):
    if frame is None:
        return None, None, rec_state

    if checkpoint_name is None:
        return None, None, rec_state

    model = load_model(checkpoint_name)

    if rec_state is None:
        rec_state = {"checkpoint": None, "rec": [None] * 4}

    # Reset temporal memory when switching checkpoints for a fair comparison.
    if rec_state.get("checkpoint") != checkpoint_name:
        rec_state = {"checkpoint": checkpoint_name, "rec": [None] * 4}

    src = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    src = src.to(DEVICE)

    with torch.no_grad():
        _, pha, *new_rec = model(src, *rec_state["rec"])

    alpha = pha[0, 0].detach().cpu().numpy().clip(0, 1)
    alpha_3 = np.repeat(alpha[:, :, None], 3, axis=2)

    if background_name:
        bg_path = BACKGROUND_DIR / background_name
        if bg_path.exists():
            bg = cv2.imread(str(bg_path))
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
            bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
        else:
            bg = np.zeros_like(frame)
    else:
        bg = np.zeros_like(frame)

    comp = frame.astype(np.float32) * alpha_3 + bg.astype(np.float32) * (1 - alpha_3)
    comp = comp.clip(0, 255).astype(np.uint8)
    alpha_vis = (alpha * 255).astype(np.uint8)

    rec_state["rec"] = new_rec
    return comp, alpha_vis, rec_state


def process_camera_tick(
    checkpoint_name: str,
    background_name: Optional[str],
    rec_state: Optional[dict],
):
    # Force Mac camera index 0 (no auto-switching).
    camera_idx = 0

    if ACTIVE_CAPTURE is None or ACTIVE_CAMERA_INDEX != camera_idx or not ACTIVE_CAPTURE.isOpened():
        open_camera(camera_idx)
        if ACTIVE_CAPTURE is None:
            return None, None, None, rec_state

    ret, frame_bgr = ACTIVE_CAPTURE.read()
    if not ret:
        return None, None, None, rec_state

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    comp, alpha_vis, rec_state = render_frame(frame_rgb, checkpoint_name, background_name, rec_state)
    return frame_rgb, comp, alpha_vis, rec_state


def build_ui() -> gr.Blocks:
    grouped_checkpoints = list_checkpoints_by_experiment()
    checkpoints = grouped_checkpoints["Pretrained"]
    backgrounds = list_backgrounds()

    css = """
    :root {
        --bg-base: #08112a;
        --bg-gradient-a: #0b1734;
        --bg-gradient-b: #0a1228;
        --panel-bg: rgba(22, 33, 64, 0.62);
        --panel-bg-2: rgba(27, 40, 76, 0.50);
        --panel-border: rgba(156, 182, 255, 0.20);
        --glass-highlight: rgba(255, 255, 255, 0.08);
        --text-primary: #eef3ff;
        --text-secondary: #aab9de;
        --text-muted: #90a1ca;
        --icon-color: #dce6ff;
        --input-bg: rgba(16, 25, 48, 0.78);
        --input-border: rgba(194, 213, 255, 0.20);
        --input-placeholder: #9caed9;
        --accent: #9ec0ff;
        --accent-2: #729eff;
        --accent-text: #071228;
        --shadow-panel: 0 14px 42px rgba(3, 9, 24, 0.54);
        --shadow-soft: 0 10px 28px rgba(10, 23, 54, 0.30);
        --overlay-glow: radial-gradient(circle at 10% 8%, rgba(132, 167, 255, 0.24) 0%, rgba(132, 167, 255, 0) 42%),
                        radial-gradient(circle at 90% 88%, rgba(112, 102, 255, 0.17) 0%, rgba(112, 102, 255, 0) 36%);
    }

    html, body, .gradio-container {
        min-height: 100vh !important;
        margin: 0 !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--text-primary) !important;
        background:
            var(--overlay-glow),
            linear-gradient(140deg, var(--bg-gradient-a) 0%, var(--bg-base) 48%, var(--bg-gradient-b) 100%) !important;
        transition: background 220ms ease, color 200ms ease;
    }

    #app-shell {
        height: calc(100vh - 18px);
        display: flex;
        flex-direction: column;
        gap: 8px;
        padding: 12px 8px 8px 8px;
        box-sizing: border-box;
    }
    #content-row {
        flex: 1 1 auto;
        min-height: 0;
        gap: 10px;
        align-items: stretch;
        flex-wrap: nowrap;
    }

    .panel {
        position: relative;
        height: 100%;
        min-height: 0;
        border: 1px solid var(--panel-border);
        border-radius: 18px;
        background: linear-gradient(170deg, var(--glass-highlight), var(--panel-bg), var(--panel-bg-2));
        box-shadow: var(--shadow-panel), inset 0 1px 0 var(--glass-highlight);
        backdrop-filter: blur(14px) saturate(118%);
        -webkit-backdrop-filter: blur(14px) saturate(118%);
        padding: 10px;
        overflow: hidden;
        transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease, background 220ms ease;
    }
    .panel:hover {
        transform: translateY(-1px);
        border-color: color-mix(in srgb, var(--panel-border) 60%, var(--accent));
        box-shadow: 0 18px 48px color-mix(in srgb, var(--bg-base) 64%, transparent), inset 0 1px 0 var(--glass-highlight);
    }

    .title {
        margin: 0;
        font-size: 20px;
        font-weight: 650;
        letter-spacing: 0.2px;
        color: var(--text-primary);
        text-shadow: 0 0 16px color-mix(in srgb, var(--accent) 24%, transparent);
    }
    .subtitle, .compact-note {
        color: var(--text-secondary);
        margin: 2px 0 0 0;
        font-size: 12px;
        letter-spacing: 0.16px;
    }
    .compact-note { margin: 0; }

    .gradio-container label,
    .gradio-container .gr-form *,
    .gradio-container .gr-group *,
    .gradio-container .gr-block-label,
    .gradio-container .gr-markdown,
    .gradio-container .gr-markdown *,
    .gradio-container .gr-text,
    .gradio-container .gr-dropdown label,
    .gradio-container .gr-image label {
        color: var(--text-primary) !important;
    }

    .gradio-container svg,
    .gradio-container .icon,
    .gradio-container [class*="icon"] {
        color: var(--icon-color) !important;
        fill: currentColor;
        stroke: currentColor;
    }

    .gr-button {
        border-radius: 11px !important;
        border: 1px solid color-mix(in srgb, var(--accent) 36%, var(--input-border)) !important;
        background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
        color: var(--accent-text) !important;
        font-weight: 650 !important;
        box-shadow: var(--shadow-soft);
        transition: transform 150ms ease, box-shadow 150ms ease, filter 150ms ease;
    }
    .gr-button:hover {
        transform: translateY(-1px);
        filter: brightness(1.02);
        box-shadow: 0 12px 30px color-mix(in srgb, var(--accent) 34%, transparent);
    }

    .gr-box, .gr-form, .gr-group, .gr-panel {
        background: transparent !important;
        border: none !important;
    }

    .gr-input, .gr-textbox, .gr-dropdown, .gradio-dropdown, .gradio-textbox,
    .gradio-container input, .gradio-container select, .gradio-container textarea {
        border-radius: 11px !important;
        border: 1px solid var(--input-border) !important;
        background: var(--input-bg) !important;
        color: var(--text-primary) !important;
        box-shadow: inset 0 1px 0 color-mix(in srgb, var(--glass-highlight) 60%, transparent);
    }

    .gradio-container input::placeholder,
    .gradio-container textarea::placeholder {
        color: var(--input-placeholder) !important;
        opacity: 1;
    }

    .gr-image, .gradio-image {
        border-radius: 14px !important;
        overflow: hidden !important;
        border: 1px solid var(--panel-border) !important;
        box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--glass-highlight) 40%, transparent);
        background: color-mix(in srgb, var(--panel-bg) 82%, transparent) !important;
    }

    .gradio-container [class*="upload"],
    .gradio-container [class*="drop"],
    .gradio-container [class*="placeholder"],
    .gradio-container [class*="toolbar"],
    .gradio-container [class*="label"],
    .gradio-container [class*="tag"],
    .gradio-container [class*="badge"] {
        color: var(--text-primary) !important;
        border-color: var(--input-border) !important;
    }

    .gr-markdown p {
        margin-top: 0.2em;
        margin-bottom: 0.2em;
    }

    @media (max-width: 1300px) {
        #content-row {
            flex-direction: column !important;
            flex-wrap: wrap;
        }
        .panel {
            min-height: 300px;
        }
    }
    """

    # Keep CSS/theme here to preserve the custom UI look.
    with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="blue")) as demo:
        with gr.Column(elem_id="app-shell"):
            gr.Markdown(
                """
                <div class="title">RVM Epoch Comparator</div>
                <p class="subtitle">Single-screen live comparison for checkpoints and backgrounds.</p>
                """
            )

            rec_state = gr.State(value={"checkpoint": None, "rec": [None] * 4})

            with gr.Row(elem_id="content-row", equal_height=False):
                with gr.Column(scale=3, min_width=260, elem_classes=["panel"]):
                    experiment_dd = gr.Dropdown(
                        label="Experiment",
                        choices=list(EXPERIMENT_DEFS.keys()),
                        value="Pretrained",
                        interactive=True,
                    )
                    experiment_desc = gr.Markdown(EXPERIMENT_DEFS["Pretrained"]["description"])
                    checkpoint_dd = gr.Dropdown(
                        label="Checkpoint / Saved Epoch",
                        choices=checkpoints,
                        value=checkpoints[0] if checkpoints else None,
                        interactive=True,
                    )
                    gr.Markdown(
                        f"<p class='compact-note'>Detected checkpoints: <b>{len(checkpoints)}</b></p>"
                    )

                    background_dd = gr.Dropdown(
                        label="Background Library",
                        choices=backgrounds,
                        value=backgrounds[0] if backgrounds else None,
                        interactive=True,
                    )
                    upload_bg = gr.Image(label="Upload Background", type="numpy", height=170)
                    save_bg_btn = gr.Button("Save Background", variant="primary")
                    save_msg = gr.Markdown()

                # All previews stacked under the main preview (single preview column).
                with gr.Column(scale=7, min_width=520, elem_classes=["panel"]):
                    composited = gr.Image(label="Composited Output", type="numpy", height=300)
                    camera_preview = gr.Image(label="Live Camera Preview", type="numpy", height=200)
                    alpha_preview = gr.Image(label="Alpha Matte", type="numpy", height=200)

        save_bg_btn.click(
            fn=save_uploaded_background,
            inputs=[upload_bg],
            outputs=[background_dd, save_msg],
        )

        tick = gr.Timer(0.05)
        tick.tick(
            fn=process_camera_tick,
            inputs=[checkpoint_dd, background_dd, rec_state],
            outputs=[camera_preview, composited, alpha_preview, rec_state],
        )

        experiment_dd.change(
            fn=checkpoints_for_experiment,
            inputs=[experiment_dd],
            outputs=[checkpoint_dd, rec_state, experiment_desc],
        )

        checkpoint_dd.change(
            fn=lambda: {"checkpoint": None, "rec": [None] * 4},
            inputs=[],
            outputs=[rec_state],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    # If GRADIO_SERVER_PORT is set, use it. Otherwise let Gradio pick a free port.
    app.launch(
        server_name="127.0.0.1",
        server_port=int(os.environ["GRADIO_SERVER_PORT"]) if os.environ.get("GRADIO_SERVER_PORT") else None,
        share=False,
    )
