import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- SAM3 ---
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ---------- metrics ----------
def to_bool_mask(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(bool)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if pred.sum() == 0 and gt.sum() == 0 else 0.0
    return float(inter / union)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0 if pred.sum() == 0 and gt.sum() == 0 else 0.0
    return float(2 * inter / denom)


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    p = float(tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if gt.sum() == 0 else 0.0)
    r = float(tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if gt.sum() == 0 else 0.0)
    return p, r


# ---------- SAM3 inference ----------
def sam3_union_mask(
    processor: Sam3Processor, img_rgb_u8: np.ndarray, prompt: str
) -> np.ndarray:
    """Return union mask (H,W) bool"""
    image_pil = Image.fromarray(img_rgb_u8, mode="RGB")
    state = processor.set_image(image_pil)
    out = processor.set_text_prompt(state=state, prompt=prompt)
    masks = out.get("masks", None)

    H, W = img_rgb_u8.shape[:2]
    if masks is None or len(masks) == 0:
        return np.zeros((H, W), dtype=bool)

    m = np.asarray(masks)
    # masks could be float/prob; make it bool
    if m.dtype != np.bool_:
        m = m > 0.5
    # union all predicted masks
    union = np.any(m.astype(bool), axis=0)
    return union


# ---------- visualization ----------
def save_viz(
    out_path: Path, img: np.ndarray, gt: np.ndarray, pred: np.ndarray, title: str
):
    """Save a 1x4 panel: image / GT / Pred / overlay"""
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    # overlay: GT in green-ish, Pred in red-ish (no fixed colors requested, but overlays need some color)
    overlay = img.copy().astype(np.float32)
    # pred highlight
    overlay[pred] = overlay[pred] * 0.4 + np.array([255, 0, 0]) * 0.6
    # gt highlight
    overlay[gt] = overlay[gt] * 0.4 + np.array([0, 255, 0]) * 0.6
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig = plt.figure(figsize=(14, 4))
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(img)
    ax1.set_title("Image")
    ax1.axis("off")
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(gt, cmap="gray")
    ax2.set_title("GT mask")
    ax2.axis("off")
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.imshow(pred, cmap="gray")
    ax3.set_title("SAM3 pred")
    ax3.axis("off")
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.imshow(overlay)
    ax4.set_title("Overlay (GT=G, Pred=R)")
    ax4.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="dir of images")
    ap.add_argument("--masks", required=True, help="dir of GT masks (png)")
    ap.add_argument("--prompt", default="hand", help='text prompt, e.g. "hand"')
    ap.add_argument(
        "--n", type=int, default=3, help="how many images to test (default 3)"
    )
    ap.add_argument("--out", default="./out_eval", help="output dir for visualizations")
    args = ap.parse_args()

    img_dir = Path(args.images)
    gt_dir = Path(args.masks)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    img_paths = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts]

    # build model once
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    count = 0
    for img_path in img_paths:
        gt_path = gt_dir / (img_path.stem + ".png")
        if not gt_path.exists():
            continue

        img = np.array(Image.open(img_path).convert("RGB"))
        gt = np.array(Image.open(gt_path).convert("L"))
        gt = to_bool_mask(gt)

        pred = sam3_union_mask(processor, img, args.prompt)

        miou = iou(pred, gt)
        mdice = dice(pred, gt)
        p, r = precision_recall(pred, gt)

        title = f"{img_path.name} | IoU={miou:.3f} Dice={mdice:.3f} P={p:.3f} R={r:.3f} | prompt='{args.prompt}'"
        print(title)

        save_viz(out_dir / f"{img_path.stem}_viz.png", img, gt, pred, title)

        count += 1
        if count >= args.n:
            break

    if count == 0:
        print(
            "No matched image/mask pairs found. Ensure masks use same stem name and are .png."
        )


if __name__ == "__main__":
    main()
