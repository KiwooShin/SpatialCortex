"""
download_run_efm3d.py — download AEO sequences and run EFM3D inference.

Downloads VRS + MPS + groundtruth for each target sequence, then runs
EFM3D inference (OBB-only, limited snippets), then runs the viz script.

Usage
-----
    python scripts/ingest/download_run_efm3d.py \
        --seqs seq03 seq04 seq05 seq06 seq07 \
        --num_snips 10
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = ROOT / "data" / "aeo"
OUT_DIR   = ROOT / "output"
VIZ_SCRIPT = ROOT / "scripts" / "viz" / "visualize_efm3d_obbs.py"
EFM3D_DIR  = Path("/home/kiwoos/efm3d")
EFM3D_INFER = EFM3D_DIR / "infer.py"
MODEL_CKPT  = EFM3D_DIR / "ckpt" / "model_release.pth"
MODEL_CFG   = EFM3D_DIR / "efm3d" / "config" / "evl_inf.yaml"

DOWNLOAD_JSON = ROOT / "AEO_download_urls.json"

SEQ_ID_MAP = {
    "seq03": "aeo_seq03_209477991816679",
    "seq04": "aeo_seq04_211239374973403",
    "seq05": "aeo_seq05_219801130654082",
    "seq06": "aeo_seq06_983146893059230",
    "seq07": "aeo_seq07_622483472741639",
}


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=2,
                  status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def download_file(url: str, dest: Path, label: str) -> None:
    """Stream-download *url* to *dest*, printing a progress bar."""
    if dest.exists():
        print(f"  SKIP {label} (already exists)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    session = _session()
    r = session.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    done  = 0
    t0    = time.time()
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
            f.write(chunk)
            done += len(chunk)
            if total:
                pct  = done / total * 100
                mb_s = done / (time.time() - t0 + 1e-6) / 1e6
                print(f"\r  {label}: {pct:5.1f}%  ({done/1e6:.0f}/{total/1e6:.0f} MB)  {mb_s:.1f} MB/s",
                      end="", flush=True)
    tmp.rename(dest)
    print(f"\r  {label}: done ({done/1e6:.0f} MB in {time.time()-t0:.0f}s)          ")


def download_and_unzip(url: str, extract_to: Path, label: str) -> None:
    """Download a zip from *url* and extract it into *extract_to*."""
    buf = io.BytesIO()
    session = _session()
    r = session.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    done  = 0
    t0    = time.time()
    for chunk in r.iter_content(chunk_size=1 << 20):
        buf.write(chunk)
        done += len(chunk)
        if total:
            pct  = done / total * 100
            mb_s = done / (time.time() - t0 + 1e-6) / 1e6
            print(f"\r  {label}: {pct:5.1f}%  ({done/1e6:.0f}/{total/1e6:.0f} MB)  {mb_s:.1f} MB/s",
                  end="", flush=True)
    buf.seek(0)
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(buf) as z:
        # Print top-level zip members for transparency
        members = z.namelist()
        print(f"\r  {label}: extracting {len(members)} files → {extract_to}")
        z.extractall(extract_to)


# ── Download one sequence ──────────────────────────────────────────────────────

def download_seq(seq_id: str, urls: dict) -> Path:
    """Download VRS, MPS, groundtruth for *seq_id*. Return sequence directory."""
    seq_dir = DATA_DIR / seq_id
    seq_dir.mkdir(parents=True, exist_ok=True)

    vrs_dest = seq_dir / "main.vrs"
    print(f"\n[{seq_id}] Downloading VRS ({urls['main_vrs']['file_size_bytes']//1e6:.0f} MB) …")
    download_file(urls["main_vrs"]["download_url"], vrs_dest, "main.vrs")

    traj_ok = any((seq_dir / "mps").rglob("closed_loop_trajectory.csv"))
    if not traj_ok:
        print(f"[{seq_id}] Downloading MPS ({urls['mps']['file_size_bytes']//1e6:.0f} MB) …")
        # Extract to seq_dir so zip's internal "mps/slam/..." lands at seq_dir/mps/slam/
        download_and_unzip(urls["mps"]["download_url"], seq_dir, "mps.zip")
    else:
        print(f"[{seq_id}] MPS already present, skipping.")

    gt_files = ["scene_objects.csv", "instances.json"]
    if not all((seq_dir / f).exists() for f in gt_files):
        print(f"[{seq_id}] Downloading groundtruth ({urls['main_groundtruth']['file_size_bytes']//1e3:.0f} KB) …")
        download_and_unzip(urls["main_groundtruth"]["download_url"], seq_dir, "groundtruth.zip")
    else:
        print(f"[{seq_id}] Groundtruth already present, skipping.")

    # Ensure trajectory is findable
    traj_candidates = list(seq_dir.rglob("closed_loop_trajectory.csv"))
    if not traj_candidates:
        raise RuntimeError(f"No closed_loop_trajectory.csv found under {seq_dir}")
    print(f"  Trajectory: {traj_candidates[0].relative_to(seq_dir)}")
    return seq_dir


# ── EFM3D inference ────────────────────────────────────────────────────────────

def run_efm3d(seq_id: str, seq_dir: Path, num_snips: int) -> Path:
    """Run EFM3D inference. Returns output dir containing snippet_obbs.csv."""
    short = seq_id[:9]   # e.g. "aeo_seq03"
    efm_out = OUT_DIR / f"efm3d_aeo_{short[4:]}"   # e.g. output/efm3d_aeo_seq03

    result_csv = efm_out / "model_release" / seq_id / "snippet_obbs.csv"
    if result_csv.exists():
        print(f"[{seq_id}] EFM3D output already exists, skipping inference.")
        return efm_out / "model_release" / seq_id

    print(f"\n[{seq_id}] Running EFM3D ({num_snips} snippets) …")
    cmd = [
        "conda", "run", "-n", "efm3d", "--no-capture-output",
        "python3", str(EFM3D_INFER),
        "--input",      str(seq_dir / "main.vrs"),  # must be .vrs file, not directory
        "--model_ckpt", str(MODEL_CKPT),
        "--model_cfg",  str(MODEL_CFG),
        "--output_dir", str(efm_out),
        "--num_snips",  str(num_snips),
        "--obb_only",
        "--skip_video",
    ]
    result = subprocess.run(cmd, cwd=str(EFM3D_DIR), check=True)
    return efm_out / "model_release" / seq_id


# ── Visualization ──────────────────────────────────────────────────────────────

def run_viz(seq_id: str, seq_dir: Path, efm_result_dir: Path, num_frames: int) -> Path:
    """Run visualize_efm3d_obbs.py. Returns output viz directory."""
    short = seq_id[:9][4:]   # "seq03"
    viz_out = OUT_DIR / f"efm3d_viz_{short}"
    if viz_out.exists() and len(list(viz_out.glob("*.jpg"))) >= num_frames:
        print(f"[{seq_id}] Viz output already exists ({len(list(viz_out.glob('*.jpg')))} frames), skipping.")
        return viz_out

    viz_out.mkdir(parents=True, exist_ok=True)
    snippet_csv = efm_result_dir / "snippet_obbs.csv"
    if not snippet_csv.exists():
        raise FileNotFoundError(f"snippet_obbs.csv not found at {snippet_csv}")

    traj_csv = next(seq_dir.rglob("closed_loop_trajectory.csv"))
    print(f"\n[{seq_id}] Running visualization …")
    cmd = [
        "conda", "run", "-n", "efm3d", "--no-capture-output",
        "python3", str(VIZ_SCRIPT),
        "--vrs",        str(seq_dir / "main.vrs"),
        "--obbs",       str(snippet_csv),
        "--traj",       str(traj_csv),
        "--output-dir", str(viz_out),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return viz_out


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=list(SEQ_ID_MAP.keys()),
                    help="Short seq names, e.g. seq03 seq04")
    ap.add_argument("--num_snips", type=int, default=10,
                    help="EFM3D snippets per sequence (each ~2 s, default 10)")
    ap.add_argument("--skip_download", action="store_true")
    ap.add_argument("--skip_inference", action="store_true")
    ap.add_argument("--skip_viz",       action="store_true")
    args = ap.parse_args()

    urls_data = json.loads(DOWNLOAD_JSON.read_text())["sequences"]

    for short in args.seqs:
        seq_id = SEQ_ID_MAP.get(short, short)
        if seq_id not in urls_data:
            print(f"  WARNING: {seq_id} not in download JSON, skipping.")
            continue

        urls    = urls_data[seq_id]
        seq_dir = DATA_DIR / seq_id

        # Step 1 — download
        if not args.skip_download:
            download_seq(seq_id, urls)
        else:
            print(f"[{seq_id}] Download skipped.")

        # Step 2 — EFM3D inference
        if not args.skip_inference:
            efm_result_dir = run_efm3d(seq_id, seq_dir, args.num_snips)
        else:
            short_name = seq_id[:9][4:]
            efm_result_dir = OUT_DIR / f"efm3d_aeo_{short_name}" / "model_release" / seq_id
            print(f"[{seq_id}] Inference skipped.")

        # Step 3 — visualization
        if not args.skip_viz:
            viz_out = run_viz(seq_id, seq_dir, efm_result_dir, num_frames=args.num_snips)
            n_frames = len(list(viz_out.glob("*.jpg")))
            print(f"[{seq_id}] Viz done → {viz_out}  ({n_frames} frames)")

    print("\n✓ All done.")


if __name__ == "__main__":
    main()
