"""
render_demo_video.py — scripted interactive-query demo video.

Assembles a ~30 s showcase of the full SpatialCortex pipeline:

  Act 1 · Title card         (2 s)
  Act 2 · EFM3D detection    (6 s)  — efm3d_viz frames with 3D OBBs
  Act 3 · Query injection    (2 s)  — "Where is the lamp?" types in
  Act 4 · Answer overlay     (3 s)  — pre-scripted answer text fades in
  Act 5 · Navigation         (10 s) — splice from existing nav MP4
  Act 6 · Target reached     (2 s)  — "TARGET REACHED" overlay

Everything rendered at 1408 × 1408 @ 24 fps.
Text rendered via Pillow (DejaVu fonts) then composited onto the canvas.

Usage
-----
    conda activate efm3d
    python scripts/viz/render_demo_video.py \\
        --out output/demo.mp4

    # Custom scene / query
    python scripts/viz/render_demo_video.py \\
        --efm3d_dir  output/efm3d_viz_seq01 \\
        --nav_video  output/nav_video_seq01_lamp_full_fixed.mp4 \\
        --nav_start  200 \\
        --nav_frames 100 \\
        --query_text "Where is the lamp? Show me the direction." \\
        --answer_line1 "The lamp is detected ~4 m ahead." \\
        --answer_line2 "Follow the arrow to navigate." \\
        --target_label "lamp · seq01" \\
        --out output/demo.mp4
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Constants ─────────────────────────────────────────────────────────────────

CANVAS     = 1408
FPS        = 24
FONT_SANS  = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD  = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO  = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# RGB tuples used for PIL; converted to BGR where needed for OpenCV
_C_BG      = ( 12,  14,  20)   # near-black background
_C_PANEL   = ( 28,  32,  44)   # card / panel fill
_C_ACCENT  = (255, 165,  30)   # warm orange accent
_C_WHITE   = (255, 255, 255)
_C_GRAY    = (170, 170, 180)
_C_GREEN   = ( 60, 210,  90)
_C_BUBBLE_Q = ( 45,  60,  90)  # user query bubble
_C_BUBBLE_A = ( 30,  45,  30)  # answer bubble


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


# ── Canvas helpers ────────────────────────────────────────────────────────────

def blank(color: tuple[int, int, int] = _C_BG) -> np.ndarray:
    """Return a CANVAS×CANVAS BGR numpy image filled with *color* (RGB input)."""
    img = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
    img[:] = color[::-1]   # RGB → BGR
    return img


def pad_to_square(bgr: np.ndarray) -> np.ndarray:
    """Centre-pad a non-square BGR image to CANVAS×CANVAS with dark fill."""
    h, w = bgr.shape[:2]
    if h == CANVAS and w == CANVAS:
        return bgr
    canvas = blank()
    y0 = (CANVAS - h) // 2
    x0 = (CANVAS - w) // 2
    canvas[y0:y0+h, x0:x0+w] = bgr
    return canvas


def darken(bgr: np.ndarray, factor: float = 0.35) -> np.ndarray:
    return np.clip(bgr.astype(np.float32) * factor, 0, 255).astype(np.uint8)


# ── PIL ↔ OpenCV bridge ───────────────────────────────────────────────────────

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ── Text helpers ──────────────────────────────────────────────────────────────

def draw_text_centered(
    bgr: np.ndarray,
    text: str,
    cy: int,
    font_path: str = FONT_SANS,
    size: int = 48,
    color: tuple = _C_WHITE,
    max_width: int | None = None,
) -> np.ndarray:
    """Draw *text* horizontally centred at vertical position *cy*."""
    pil = bgr_to_pil(bgr)
    d   = ImageDraw.Draw(pil)
    fnt = _font(font_path, size)
    bb  = d.textbbox((0, 0), text, font=fnt)
    tw  = bb[2] - bb[0]
    th  = bb[3] - bb[1]
    x   = (CANVAS - tw) // 2
    y   = cy - th // 2
    if max_width and tw > max_width:
        # scale down font and recurse
        scale = max_width / tw
        return draw_text_centered(bgr, text, cy, font_path,
                                  int(size * scale), color, None)
    d.text((x, y), text, font=fnt, fill=color)
    return pil_to_bgr(pil)


def draw_text_left(
    bgr: np.ndarray,
    text: str,
    xy: tuple[int, int],
    font_path: str = FONT_SANS,
    size: int = 36,
    color: tuple = _C_WHITE,
) -> np.ndarray:
    pil = bgr_to_pil(bgr)
    d   = ImageDraw.Draw(pil)
    fnt = _font(font_path, size)
    d.text(xy, text, font=fnt, fill=color)
    return pil_to_bgr(pil)


def label_strip(bgr: np.ndarray, text: str, color: tuple = _C_GRAY) -> np.ndarray:
    """Draw a small label in the bottom-left corner."""
    return draw_text_left(bgr, text, (24, CANVAS - 48), FONT_SANS, 30, color)


def step_badge(bgr: np.ndarray, step: str) -> np.ndarray:
    """Draw a 'Step N · <label>' badge in the top-left corner."""
    pil = bgr_to_pil(bgr)
    d   = ImageDraw.Draw(pil)
    fnt = _font(FONT_BOLD, 28)
    pad = 10
    bb  = d.textbbox((0, 0), step, font=fnt)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    d.rounded_rectangle([16, 16, 16 + tw + pad*2, 16 + th + pad*2],
                        radius=8, fill=_C_PANEL)
    d.text((16 + pad, 16 + pad), step, font=fnt, fill=_C_ACCENT)
    return pil_to_bgr(pil)


# ── Chat-bubble overlay ───────────────────────────────────────────────────────

def draw_bubble(
    bgr: np.ndarray,
    text: str,
    cy: int,
    role: str = "user",   # "user" or "assistant"
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Draw a rounded chat bubble centered at *cy*.
    role="user"      → right-aligned, query style
    role="assistant" → left-aligned, answer style
    *alpha* fades the bubble and text in (0.0 = invisible, 1.0 = fully visible).
    """
    pil = bgr_to_pil(bgr)
    overlay = pil.copy()
    d = ImageDraw.Draw(overlay)
    fnt = _font(FONT_SANS, 42)

    lines = text.split("\n")
    line_bbs = [d.textbbox((0, 0), ln, font=fnt) for ln in lines]
    line_h = max(bb[3] - bb[1] for bb in line_bbs) + 4
    max_w  = max(bb[2] - bb[0] for bb in line_bbs)
    total_h = line_h * len(lines)

    pad_x, pad_y = 36, 24
    bw = max_w + pad_x * 2
    bh = total_h + pad_y * 2
    margin = 80

    if role == "user":
        bx2 = CANVAS - margin
        bx1 = bx2 - bw
        tc = _C_WHITE
        bc = _C_BUBBLE_Q
    else:
        bx1 = margin
        bx2 = bx1 + bw
        tc = _C_GREEN
        bc = _C_BUBBLE_A

    by1 = cy - bh // 2
    by2 = by1 + bh

    d.rounded_rectangle([bx1, by1, bx2, by2], radius=16, fill=bc)
    for i, (ln, bb) in enumerate(zip(lines, line_bbs)):
        lw = bb[2] - bb[0]
        if role == "user":
            tx = bx2 - pad_x - lw
        else:
            tx = bx1 + pad_x
        ty = by1 + pad_y + i * line_h
        d.text((tx, ty), ln, font=fnt, fill=tc)

    if alpha < 1.0:
        pil = Image.blend(pil, overlay, alpha)
    else:
        pil = overlay

    return pil_to_bgr(pil)


# ── Writer helper ─────────────────────────────────────────────────────────────

def make_writer(path: str) -> cv2.VideoWriter:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (CANVAS, CANVAS),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {path}")
    return writer


def write_n(writer: cv2.VideoWriter, frame: np.ndarray, n: int) -> None:
    for _ in range(n):
        writer.write(frame)


# ── Acts ──────────────────────────────────────────────────────────────────────

def act_title(writer: cv2.VideoWriter, secs: float = 2.0) -> None:
    """Dark title card with project name and pipeline overview."""
    n = round(secs * FPS)
    base = blank()

    # Accent rule lines
    pil = bgr_to_pil(base)
    d   = ImageDraw.Draw(pil)
    rule_y = CANVAS // 2 - 10
    d.line([(120, rule_y - 90), (CANVAS - 120, rule_y - 90)], fill=_C_ACCENT, width=2)
    d.line([(120, rule_y + 140), (CANVAS - 120, rule_y + 140)], fill=_C_ACCENT, width=2)
    base = pil_to_bgr(pil)

    base = draw_text_centered(base, "SpatialCortex",
                               CANVAS // 2 - 50, FONT_BOLD, 110, _C_WHITE,
                               max_width=CANVAS - 160)
    base = draw_text_centered(base,
                               "Spatial Object Retrieval & Navigation on Project Aria",
                               CANVAS // 2 + 60, FONT_SANS, 38, _C_GRAY,
                               max_width=CANVAS - 160)
    base = draw_text_centered(base,
                               "Demo: Natural Language Query  →  3D Navigation",
                               CANVAS // 2 + 160, FONT_SANS, 34, _C_ACCENT,
                               max_width=CANVAS - 160)

    write_n(writer, base, n)
    print(f"  [act_title]  {n} frames")


def act_detect(
    writer: cv2.VideoWriter,
    efm3d_dir: str,
    secs: float = 6.0,
) -> np.ndarray:
    """Play EFM3D detection frames padded to square. Returns last frame."""
    frames_paths = sorted(Path(efm3d_dir).glob("*.jpg"))
    if not frames_paths:
        raise FileNotFoundError(f"No .jpg frames in {efm3d_dir}")

    n_out  = round(secs * FPS)
    repeat = max(1, round(n_out / len(frames_paths)))
    last   = None

    last_raw = None
    for fp in frames_paths:
        raw  = cv2.imread(str(fp))
        if raw is None:
            continue
        padded   = pad_to_square(raw)
        last_raw = padded           # save pre-badge version for query/answer bg
        frm      = step_badge(padded, "Step 1 · EFM3D 3D Object Detection")
        frm      = label_strip(frm, Path(fp).stem, _C_GRAY)
        write_n(writer, frm, repeat)

    written = len(frames_paths) * repeat
    print(f"  [act_detect]  {len(frames_paths)} source frames × {repeat} → {written} frames")
    return last_raw  # type: ignore[return-value]


def act_query(
    writer: cv2.VideoWriter,
    bg: np.ndarray,
    query_text: str,
    secs: float = 2.5,
) -> None:
    """Typing animation: query text appears character-by-character."""
    n     = round(secs * FPS)
    dark  = darken(bg, 0.4)
    dark  = step_badge(dark, "Step 2 · Query Processing")
    chars = len(query_text)

    for i in range(n):
        t   = i / n
        # How many characters to show (ease-in, then hold)
        shown = min(chars, math.ceil(t * chars * 1.2))
        cursor = "|" if (i % 14) < 9 else " "
        partial = query_text[:shown] + cursor

        frm = draw_bubble(dark.copy(), partial, cy=CANVAS // 2, role="user")
        writer.write(frm)

    print(f"  [act_query]   {n} frames (typing '{query_text[:20]}...')")


def act_answer(
    writer: cv2.VideoWriter,
    bg: np.ndarray,
    query_text: str,
    answer_line1: str,
    answer_line2: str,
    secs: float = 3.0,
) -> None:
    """Query bubble stays; answer bubble fades in below."""
    n          = round(secs * FPS)
    dark       = darken(bg, 0.4)
    dark       = step_badge(dark, "Step 2 · CLIP Retrieval → Located")
    query_full = query_text
    answer     = f"{answer_line1}\n{answer_line2}"
    fade_frames = round(0.6 * FPS)

    for i in range(n):
        alpha = min(1.0, i / fade_frames)
        frm   = draw_bubble(dark.copy(), query_full,
                            cy=CANVAS // 2 - 160, role="user")
        frm   = draw_bubble(frm, answer,
                            cy=CANVAS // 2 + 140, role="assistant", alpha=alpha)
        writer.write(frm)

    print(f"  [act_answer]  {n} frames")


def act_nav(
    writer: cv2.VideoWriter,
    nav_video: str,
    secs: float = 10.0,
    start_frame: int = 200,
) -> np.ndarray:
    """Splice a segment from the pre-rendered nav video. Returns last frame."""
    cap     = cv2.VideoCapture(nav_video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    n_src   = round(secs * src_fps)          # source frames to read
    n_out   = round(secs * FPS)              # output frames to write
    repeat  = max(1, round(FPS / src_fps))   # output frames per source frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames_read = 0
    last = None

    while frames_read < n_src:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[0] != CANVAS or frame.shape[1] != CANVAS:
            frame = cv2.resize(frame, (CANVAS, CANVAS))
        frm  = step_badge(frame, "Step 3 · Visual Navigation")
        last = frm
        write_n(writer, frm, repeat)
        frames_read += 1

    cap.release()
    written = frames_read * repeat
    print(f"  [act_nav]     {frames_read} source frames × {repeat} → {written} frames")
    return last  # type: ignore[return-value]


def act_arrived(
    writer: cv2.VideoWriter,
    bg: np.ndarray,
    target_label: str,
    secs: float = 2.0,
) -> None:
    """Final 'TARGET REACHED' card over the last nav frame."""
    n    = round(secs * FPS)
    dark = np.clip(bg.astype(np.float32) * 0.55, 0, 255).astype(np.uint8)
    dark = draw_text_centered(dark, "TARGET REACHED",
                               CANVAS // 2 - 40, FONT_BOLD, 96, _C_GREEN,
                               max_width=CANVAS - 160)
    dark = draw_text_centered(dark, target_label,
                               CANVAS // 2 + 80, FONT_SANS, 44, _C_GRAY,
                               max_width=CANVAS - 200)
    write_n(writer, dark, n)
    print(f"  [act_arrived] {n} frames")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Render SpatialCortex interactive-query demo video")
    ap.add_argument("--out",         default="output/demo.mp4")
    ap.add_argument("--efm3d_dir",   default="output/efm3d_viz_seq01")
    ap.add_argument("--nav_video",   default="output/nav_video_seq01_lamp_full_fixed.mp4")
    ap.add_argument("--nav_start",   type=int,   default=200,
                    help="Start frame in the nav video (default 200 = 20 s in)")
    ap.add_argument("--nav_secs",    type=float, default=10.0)
    ap.add_argument("--query_text",  default="Where is the lamp? Show me the direction.")
    ap.add_argument("--answer_line1",default="The lamp is detected ~4 m ahead.")
    ap.add_argument("--answer_line2",default="Follow the arrow to navigate.")
    ap.add_argument("--target_label",default="lamp · seq01")
    args = ap.parse_args()

    print(f"Output : {args.out}")
    print(f"Canvas : {CANVAS}×{CANVAS} @ {FPS} fps\n")

    writer = make_writer(args.out)

    # Act 1 — Title
    print("Rendering Act 1: Title …")
    act_title(writer, secs=2.0)

    # Act 2 — Detection
    print("Rendering Act 2: Detection …")
    last_detect = act_detect(writer, args.efm3d_dir, secs=6.0)

    # Act 3 — Query typing
    print("Rendering Act 3: Query …")
    act_query(writer, last_detect, args.query_text, secs=2.5)

    # Act 4 — Answer
    print("Rendering Act 4: Answer …")
    act_answer(writer, last_detect, args.query_text,
               args.answer_line1, args.answer_line2, secs=3.0)

    # Act 5 — Navigation
    print("Rendering Act 5: Navigation …")
    last_nav = act_nav(writer, args.nav_video,
                       secs=args.nav_secs, start_frame=args.nav_start)

    # Act 6 — Arrived
    print("Rendering Act 6: Arrived …")
    if last_nav is not None:
        act_arrived(writer, last_nav, args.target_label, secs=2.0)
    else:
        act_arrived(writer, blank(), args.target_label, secs=2.0)

    writer.release()

    from spatialcortex.config import OUTPUT_DIR
    total_secs = 2.0 + 6.0 + 2.5 + 3.0 + args.nav_secs + 2.0
    print(f"\n✓ Demo video saved to: {args.out}")
    print(f"  Approx duration: {total_secs:.0f} s")


if __name__ == "__main__":
    main()
