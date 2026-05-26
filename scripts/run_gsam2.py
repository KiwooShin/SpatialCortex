#!/usr/bin/env python3
"""
SpatialCortex — Grounded SAM 2 detection on COLMAP-registered frames.

Runs Grounding DINO (open-vocabulary detection) + SAM 2 (segmentation) on
every frame registered in a COLMAP reconstruction.  Outputs per-frame JSON
detection files and binary mask PNGs consumed by lift_to_3d.py.

Output layout:
    <output>/
    ├── all_detections.json          ← dict keyed by frame name
    ├── frame_00042_boxes.json       ← [{label, confidence, bbox_2d, ...}]
    └── frame_00042_masks/
        ├── toaster_0.png
        └── dish_1.png

Usage:
    conda activate gsam2
    python scripts/run_gsam2.py \\
        --images        data/colmap/images \\
        --colmap-images data/colmap/sparse/0/images.txt \\
        --output        data/detections \\
        --sam2-checkpoint   ~/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt \\
        --sam2-config       configs/sam2.1/sam2.1_hiera_l.yaml \\
        --gdino-checkpoint  ~/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth \\
        --gdino-config      ~/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py

Setup (one-time):
    conda create -n gsam2 python=3.10 -y && conda activate gsam2
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    git clone https://github.com/IDEA-Research/Grounded-SAM-2 ~/Grounded-SAM-2
    cd ~/Grounded-SAM-2 && pip install -e . && pip install -e grounding_dino
    pip install pillow tqdm numpy opencv-python

    # Weights
    cd ~/Grounded-SAM-2
    wget -P checkpoints/ \\
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
    wget -P gdino_checkpoints/ \\
        https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ── Prompt ────────────────────────────────────────────────────────────────────

KITCHEN_PROMPT = (
    "dish . toaster . sink . faucet . stove . cooker . mug . bowl . cup . "
    "bottle . knife . pot . pan . plate . glass . spoon . fork . kettle . "
    "microwave . refrigerator . cabinet . cutting board"
)
BOX_THRESHOLD  = 0.30
TEXT_THRESHOLD = 0.25


# ── COLMAP helpers ────────────────────────────────────────────────────────────

def load_registered_names(images_txt: Path) -> set:
    """Return the set of image filenames registered by COLMAP."""
    names = set()
    with open(images_txt) as f:
        skip_obs = False
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            if not skip_obs:
                parts = line.split()
                if len(parts) >= 10:
                    names.add(parts[9])
            skip_obs = not skip_obs
    return names


# ── Model loading ─────────────────────────────────────────────────────────────

def build_gdino(config_path: str, checkpoint_path: str, device: str):
    from grounding_dino.util.slconfig import SLConfig
    from grounding_dino.models import build_model as _build
    from grounding_dino.util.utils import clean_state_dict

    cfg = SLConfig.fromfile(config_path)
    cfg.device = device
    model = _build(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    model.eval().to(device)
    return model


def build_sam2(config_path: str, checkpoint_path: str, device: str):
    from sam2.build_sam import build_sam2 as _build
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = _build(config_path, checkpoint_path, device=device)
    return SAM2ImagePredictor(model)


# ── Inference ─────────────────────────────────────────────────────────────────

def gdino_transform():
    import groundingdino.datasets.transforms as T
    return T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def run_gdino(model, transform, image_pil: Image.Image,
              prompt: str, box_thresh: float, text_thresh: float,
              device: str):
    from grounding_dino.util.utils import get_phrases_from_posmap

    img_tensor, _ = transform(image_pil, None)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor[None], captions=[prompt])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]   # (N, num_tokens)
    boxes  = outputs["pred_boxes"].cpu()[0]              # (N, 4) cxcywh norm

    filt = logits.max(dim=1)[0] > box_thresh
    logits_filt = logits[filt]
    boxes_filt  = boxes[filt]

    tokenized = model.tokenizer(prompt)
    phrases = []
    for row in logits_filt:
        phrase = get_phrases_from_posmap(row > text_thresh, tokenized, model.tokenizer)
        confidence = float(row.max().item())
        phrases.append((phrase.strip(), confidence))

    return boxes_filt, phrases  # both (M, *), M ≤ N


def boxes_to_xyxy(boxes_cxcywh: torch.Tensor, W: int, H: int) -> np.ndarray:
    """Convert normalized cxcywh boxes to pixel xyxy."""
    cx, cy, bw, bh = boxes_cxcywh.unbind(dim=1)
    x1 = (cx - bw / 2) * W
    y1 = (cy - bh / 2) * H
    x2 = (cx + bw / 2) * W
    y2 = (cy + bh / 2) * H
    return torch.stack([x1, y1, x2, y2], dim=1).numpy()


# ── Per-frame processing ──────────────────────────────────────────────────────

def process_frame(img_path: Path, gdino_model, gdino_tf, sam2_predictor,
                  prompt: str, box_thresh: float, text_thresh: float,
                  out_dir: Path, device: str) -> list:
    img_pil = Image.open(img_path).convert("RGB")
    W, H = img_pil.size

    boxes_norm, phrases = run_gdino(
        gdino_model, gdino_tf, img_pil, prompt,
        box_thresh, text_thresh, device,
    )

    if len(boxes_norm) == 0:
        return []

    boxes_xyxy = boxes_to_xyxy(boxes_norm, W, H)
    # clip to image bounds
    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, W)
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, H)

    # SAM 2
    sam2_predictor.set_image(np.array(img_pil))
    masks, scores, _ = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes_xyxy,
        multimask_output=False,
    )
    if masks.ndim == 4:          # (N, 1, H, W)
        masks = masks[:, 0]      # → (N, H, W)

    # Save masks + build detection records
    stem      = img_path.stem
    masks_dir = out_dir / f"{stem}_masks"
    masks_dir.mkdir(exist_ok=True)

    detections = []
    for i, ((label, conf), box, mask, score) in enumerate(
        zip(phrases, boxes_xyxy, masks, scores)
    ):
        safe_label  = label.replace(" ", "_") if label else "object"
        mask_bin    = (mask > 0.5).astype(np.uint8) * 255
        mask_fname  = f"{safe_label}_{i}.png"
        Image.fromarray(mask_bin).save(masks_dir / mask_fname)

        detections.append({
            "label":      label or "object",
            "confidence": round(conf, 4),
            "sam2_score": round(float(score), 4),
            "bbox_2d":    [round(float(v), 1) for v in box],
            "mask_path":  str((masks_dir / mask_fname).relative_to(out_dir.parent)),
            "mask_area":  int(mask_bin.sum()) // 255,
        })

    return detections


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run Grounded SAM 2 on COLMAP-registered frames."
    )
    parser.add_argument("--images",        required=True,
                        help="Directory of COLMAP images")
    parser.add_argument("--colmap-images", required=True,
                        help="Path to sparse/0/images.txt")
    parser.add_argument("--output",        required=True,
                        help="Output directory for detections")
    parser.add_argument("--sam2-checkpoint",
                        default="~/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-config",
                        default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--gdino-checkpoint",
                        default="~/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth")
    parser.add_argument("--gdino-config",
                        default="~/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--prompt",        default=KITCHEN_PROMPT)
    parser.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD)
    parser.add_argument("--text-threshold",type=float, default=TEXT_THRESHOLD)
    parser.add_argument("--device",        default="cuda")
    args = parser.parse_args()

    sam2_ckpt   = os.path.expanduser(args.sam2_checkpoint)
    gdino_ckpt  = os.path.expanduser(args.gdino_checkpoint)
    gdino_cfg   = os.path.expanduser(args.gdino_config)

    for p, label in [(sam2_ckpt, "SAM2 checkpoint"), (gdino_ckpt, "GDINO checkpoint"),
                     (gdino_cfg, "GDINO config")]:
        if not Path(p).exists():
            print(f"Error: {label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    out_dir   = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir   = Path(args.images)
    registered = load_registered_names(Path(args.colmap_images))
    print(f"COLMAP-registered frames: {len(registered)}")

    print("Loading Grounding DINO …")
    gdino_model = build_gdino(gdino_cfg, gdino_ckpt, args.device)
    gdino_tf    = gdino_transform()

    print("Loading SAM 2 …")
    sam2_pred = build_sam2(args.sam2_config, sam2_ckpt, args.device)

    img_files = sorted(
        f for f in img_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png") and f.name in registered
    )
    print(f"Processing {len(img_files)} registered frames …")

    all_detections = {}
    for img_path in tqdm(img_files):
        dets = process_frame(
            img_path, gdino_model, gdino_tf, sam2_pred,
            args.prompt, args.box_threshold, args.text_threshold,
            out_dir, args.device,
        )
        all_detections[img_path.name] = dets

        if dets:
            stem = img_path.stem
            (out_dir / f"{stem}_boxes.json").write_text(
                json.dumps(dets, indent=2)
            )

    (out_dir / "all_detections.json").write_text(
        json.dumps(all_detections, indent=2)
    )

    total_dets = sum(len(v) for v in all_detections.values())
    frames_with = sum(1 for v in all_detections.values() if v)
    print(f"\nDone.  {total_dets} detections across {frames_with}/{len(img_files)} frames.")
    print(f"Output → {out_dir}/")


if __name__ == "__main__":
    main()
