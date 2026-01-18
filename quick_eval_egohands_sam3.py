import argparse
from pathlib import Path
import random
import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 你这份数据的帧名明显是 frame_####.jpg（从你运行结果看出来）
# 但我仍保留几种兜底规则
CANDIDATES = [
    "{video}/frame_{frame}.jpg",
    "{video}/frame_{frame}.png",
    "{video}/frame_{frame:04d}.jpg",
    "{video}/frame_{frame:04d}.png",
    "{video}/frame_{frame:06d}.jpg",
    "{video}/frame_{frame:06d}.png",
    "{video}/{frame}.jpg",
    "{video}/{frame}.png",
    "{video}/{frame:06d}.jpg",
    "{video}/{frame:06d}.png",
]


def polygon_to_mask(hw, poly_xy):
    if poly_xy is None or len(poly_xy) == 0:
        return None
    H, W = hw
    m = Image.new("L", (W, H), 0)
    pts = [(float(x), float(y)) for x, y in poly_xy]
    ImageDraw.Draw(m).polygon(pts, outline=1, fill=1)
    return np.array(m, dtype=bool)


def find_frame_image(root: Path, video_id: str, frame_num: int) -> Path | None:
    base = root / "_LABELLED_SAMPLES"
    for pat in CANDIDATES:
        rel = pat.format(video=video_id, frame=frame_num)
        p = base / rel
        if p.exists():
            return p
    vdir = base / video_id
    if vdir.exists():
        # 兜底：按 frame_num 搜
        for p in vdir.glob("*"):
            if (
                p.suffix.lower() in [".jpg", ".jpeg", ".png"]
                and str(frame_num) in p.stem
            ):
                return p
    return None


def sam3_predict(
    processor: Sam3Processor, img_u8: np.ndarray, prompt: str, merge: str
) -> np.ndarray:
    """return bool mask HxW"""
    image_pil = Image.fromarray(img_u8, mode="RGB")
    state = processor.set_image(image_pil)
    out = processor.set_text_prompt(state=state, prompt=prompt)

    masks = out.get("masks", None)
    H, W = img_u8.shape[:2]
    if masks is None or len(masks) == 0:
        return np.zeros((H, W), dtype=bool)

    masks = np.asarray(masks).astype(bool)  # (N,H,W)

    if merge == "union":
        return np.any(masks, axis=0)

    # best: pick highest score if provided
    scores = out.get("scores", None)
    if scores is not None and len(scores) == len(masks):
        best_idx = int(np.argmax(np.asarray(scores)))
        return masks[best_idx]

    return np.any(masks, axis=0)


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
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    p = float(tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if gt.sum() == 0 else 0.0)
    r = float(tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if gt.sum() == 0 else 0.0)
    return p, r


def overlay(img_u8, mask, alpha=0.45, color=(255, 0, 0)):
    img = img_u8.astype(np.float32)
    color = np.array(color, dtype=np.float32)[None, None, :]
    m3 = np.repeat(mask[:, :, None], 3, axis=2)
    out = np.where(m3, (1 - alpha) * img + alpha * color, img)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--egohands_root",
        required=True,
        help="contains metadata.mat and _LABELLED_SAMPLES/",
    )
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="hand")
    ap.add_argument(
        "--merge",
        choices=["best", "union"],
        default="union",
        help="how to merge SAM3 multiple masks; union is safer for 'hand' concept",
    )
    ap.add_argument("--out", default="./sam3_eval_out")
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.egohands_root)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # load metadata
    m = sio.loadmat(root / "metadata.mat", squeeze_me=True, struct_as_record=False)
    videos = (
        list(m["video"].flat) if isinstance(m["video"], np.ndarray) else [m["video"]]
    )

    refs = []
    for vid in videos:
        video_id = str(getattr(vid, "video_id"))
        lf = getattr(vid, "labelled_frames")
        lf_list = list(lf.flat) if isinstance(lf, np.ndarray) else [lf]
        for fr in lf_list:
            refs.append((video_id, fr))
    print("Total labelled frames:", len(refs))

    picks = random.sample(refs, k=min(args.n, len(refs)))

    # init SAM3 once
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    all_iou, all_dice = [], []
    for idx, (video_id, fr) in enumerate(picks):
        frame_num = int(getattr(fr, "frame_num"))
        img_path = find_frame_image(root, video_id, frame_num)
        if img_path is None:
            print(f"[{idx}] NOT FOUND image for {video_id} frame={frame_num}")
            continue

        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        H, W = img.shape[:2]

        # GT: union of 4 hands
        union_gt = np.zeros((H, W), dtype=bool)
        for name in ["myleft", "myright", "yourleft", "yourright"]:
            poly = getattr(fr, name)
            if isinstance(poly, np.ndarray) and poly.size > 0:
                msk = polygon_to_mask((H, W), poly)
                if msk is not None:
                    union_gt |= msk

        # SAM3 pred
        pred = sam3_predict(processor, img, args.prompt, merge=args.merge)

        miou = iou(pred, union_gt)
        mdice = dice(pred, union_gt)
        p, r = precision_recall(pred, union_gt)

        all_iou.append(miou)
        all_dice.append(mdice)

        stem = f"{video_id}_f{frame_num:06d}"
        # save images
        Image.fromarray(img).save(outdir / f"{stem}_img.png")
        Image.fromarray((union_gt.astype(np.uint8) * 255), mode="L").save(
            outdir / f"{stem}_gt.png"
        )
        Image.fromarray((pred.astype(np.uint8) * 255), mode="L").save(
            outdir / f"{stem}_pred.png"
        )

        overlay(img, union_gt, color=(0, 255, 0)).save(
            outdir / f"{stem}_overlay_gt_green.png"
        )
        overlay(img, pred, color=(255, 0, 0)).save(
            outdir / f"{stem}_overlay_pred_red.png"
        )

        # diff 可视化：TP=白, FP=红, FN=蓝
        tp = np.logical_and(pred, union_gt)
        fp = np.logical_and(pred, ~union_gt)
        fn = np.logical_and(~pred, union_gt)
        diff = np.zeros((H, W, 3), dtype=np.uint8)
        diff[tp] = (255, 255, 255)
        diff[fp] = (255, 0, 0)
        diff[fn] = (0, 0, 255)
        Image.fromarray(diff).save(outdir / f"{stem}_diff_tpwhite_fpred_fnblue.png")

        print(f"[{idx}] {stem}: IoU={miou:.4f} Dice={mdice:.4f} P={p:.4f} R={r:.4f}")

    if all_iou:
        print(
            f"\nMean over {len(all_iou)} samples: mIoU={float(np.mean(all_iou)):.4f}  mDice={float(np.mean(all_dice)):.4f}"
        )
    print("Saved to:", outdir.resolve())


if __name__ == "__main__":
    main()
# python quick_eval_egohands_sam3.py --egohands_root egohands --n 5 --prompt "hand" --merge union --out ./sam3_egohands_eval_out
