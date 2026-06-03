"""
ODAOCR — standalone inference script for Radio Barcelona line crops.

Loads the meta-learnt checkpoint and runs handwritten OCR on one or more
line-crop images.  Includes projection-profile word segmentation: each line
crop is split into individual word crops before recognition, then the results
are joined.  This avoids the accuracy loss that occurs when the model tries to
read a full-width line squashed into 224×224 pixels.

Usage
-----
# single image
python infer.py path/to/line_crop.jpg

# directory of crops (all .jpg / .png)
python infer.py path/to/crops/

# explicit model / tokenizer locations
python infer.py crops/ \\
    --checkpoint MODELS/model_metalearnt.pt \\
    --tokenizer-dir MODELS \\
    --device cuda

Output
------
One line per image:  <filename>\\t<transcription>

Requirements
------------
pip install torch torchvision pillow timm opencv-python-headless
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ODAOCR lives alongside this script — add its directory to the path so that
# constructors.py and vitstr.py are importable regardless of the working dir.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from constructors import prepare_model, make_inference, GreedyTextDecoder, CharTokenizer

SCRIPT_DIR  = Path(__file__).resolve().parent
DEFAULT_CKPT = SCRIPT_DIR / "MODELS" / "model_metalearnt.pt"
DEFAULT_TOKDIR = SCRIPT_DIR / "MODELS"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ── word segmentation via vertical ink projection ─────────────────────────────

def split_words(crop_bgr: np.ndarray,
                min_gap: int = 8,
                min_word_w: int = 10) -> list[np.ndarray]:
    """
    Split a line crop into word-sized sub-crops using a column-wise ink
    projection profile.

    A column that has at least one dark pixel is considered "inked".  Runs of
    inked columns form word candidates; gaps narrower than `min_gap` pixels
    are merged so that letters within one word are not split.

    Parameters
    ----------
    crop_bgr   : BGR image (H×W×3)
    min_gap    : gaps smaller than this (px) are merged
    min_word_w : segments narrower than this are discarded (noise)

    Returns a list of word crops in reading order.  If nothing is found the
    original crop is returned as a single-element list.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    proj = ((255 - binary) // 255).sum(axis=0).astype(int)   # ink count per column

    w = crop_bgr.shape[1]
    in_word, segs, x0 = False, [], 0
    for x in range(w):
        if proj[x] > 0:
            if not in_word:
                x0, in_word = x, True
        else:
            if in_word:
                in_word = False
                if x - x0 >= min_word_w:
                    segs.append([x0, x])
    if in_word and w - x0 >= min_word_w:
        segs.append([x0, w])

    # merge gaps smaller than min_gap
    merged: list[list[int]] = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < min_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)

    if not merged:
        return [crop_bgr]

    pad = 4
    return [crop_bgr[:, max(0, a - pad):min(w, b + pad)] for a, b in merged]


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint: Path, tokenizer_dir: Path, device: str):
    """Return (model, tokenizer, decoder) ready for inference."""
    tokenizer = CharTokenizer(False, str(tokenizer_dir), "tokenizer")
    model = prepare_model(
        len(tokenizer),
        device=device,
        load_checkpoint=True,
        checkpoint_name=str(checkpoint),
    )
    model.eval()
    return model, tokenizer, GreedyTextDecoder()


# ── inference helpers ─────────────────────────────────────────────────────────

def infer_single(crop_bgr: np.ndarray, model, tokenizer, decoder,
                 device: str) -> str:
    """Run the model on one (possibly small) crop and return the string."""
    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    results = make_inference(model, tokenizer, decoder, pil, device)
    return results[0].strip() if results else ""


def infer_line(crop_bgr: np.ndarray, model, tokenizer, decoder,
               device: str) -> str:
    """
    Word-split a line crop, run inference on each word, return joined text.
    Falls back to whole-line inference if segmentation yields nothing.
    """
    words = split_words(crop_bgr)
    parts = [infer_single(wc, model, tokenizer, decoder, device) for wc in words]
    return " ".join(p for p in parts if p)


# ── CLI ───────────────────────────────────────────────────────────────────────

def collect_images(paths: list[Path]) -> list[Path]:
    imgs = []
    for path in paths:
        if path.is_dir():
            found = sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS)
            if not found:
                print(f"Warning: no images found in {path}", file=sys.stderr)
            imgs.extend(found)
        elif path.is_file() and path.suffix.lower() in IMG_EXTS:
            imgs.append(path)
        else:
            sys.exit(f"Not an image file or directory: {path}")
    if not imgs:
        sys.exit("No images to process.")
    return imgs


def get_args():
    p = argparse.ArgumentParser(
        description="ODAOCR inference with word-split segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", type=Path, nargs="+",
                   help="One or more image files or directories of line-crop images")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT,
                   help=f"Model checkpoint (default: {DEFAULT_CKPT})")
    p.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKDIR,
                   help=f"Directory containing tokenizer.json (default: {DEFAULT_TOKDIR})")
    p.add_argument("--device", default="cuda",
                   help="Torch device: cuda or cpu (default: cuda)")
    p.add_argument("--min-gap", type=int, default=8,
                   help="Word-split: minimum gap between words in pixels (default: 8)")
    p.add_argument("--min-word-w", type=int, default=10,
                   help="Word-split: minimum word width in pixels (default: 10)")
    return p.parse_args()


def main():
    args = get_args()

    if not args.checkpoint.exists():
        sys.exit(
            f"Checkpoint not found: {args.checkpoint}\n"
            f"Run down.sh to download the model weights."
        )
    tok_file = args.tokenizer_dir / "tokenizer.json"
    if not tok_file.exists():
        sys.exit(
            f"tokenizer.json not found in {args.tokenizer_dir}\n"
            f"Run down.sh to download and unzip the tokenizer."
        )

    images = collect_images(args.input)
    print(f"Loading model from {args.checkpoint} on {args.device} …",
          file=sys.stderr)
    model, tokenizer, decoder = load_model(
        args.checkpoint, args.tokenizer_dir, args.device
    )
    print(f"Running inference on {len(images)} image(s) …\n", file=sys.stderr)

    for img_path in images:
        crop = cv2.imread(str(img_path))
        if crop is None:
            print(f"{img_path.name}\t[error: could not read image]")
            continue
        text = infer_line(crop, model, tokenizer, decoder, args.device)
        print(f"{img_path.name}\t{text}")


if __name__ == "__main__":
    main()
