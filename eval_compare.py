import os, argparse, glob
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from predictor_stub import NoModelPredictor  # 先用占位跑通

# 以后你接SAM3就改成：from predictor_stub import Sam3Predictor


def load_mask(path):
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def mask_to_bbox(mask01):
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def iou(pred01, gt01):
    pred = pred01.astype(bool)
    gt = gt01.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / float(union + 1e-9)


def precision_recall(pred01, gt01):
    pred = pred01.astype(bool)
    gt = gt01.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    prec = float(tp) / float(tp + fp + 1e-9)
    rec = float(tp) / float(tp + fn + 1e-9)
    return prec, rec


def overlay(img_bgr, gt01, pred01):
    img = img_bgr.copy()
    gt_c = np.zeros_like(img)
    gt_c[..., 1] = gt01 * 255  # green
    pr_c = np.zeros_like(img)
    pr_c[..., 2] = pred01 * 255  # red
    mix = cv2.addWeighted(img, 1.0, gt_c, 0.35, 0)
    mix = cv2.addWeighted(mix, 1.0, pr_c, 0.35, 0)
    return mix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out_dir", default="out_eval")
    ap.add_argument("--num_viz", type=int, default=3)
    ap.add_argument("--baseline_mode", choices=["bbox", "empty"], default="bbox")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    viz_dir = os.path.join(args.out_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(args.images_dir, "*.*")))
    if not img_paths:
        raise RuntimeError("images_dir 里没找到图片")

    predictor = NoModelPredictor(mode=args.baseline_mode)
    rows = []
    viz_saved = 0

    for ip in tqdm(img_paths):
        stem = os.path.splitext(os.path.basename(ip))[0]
        gp = os.path.join(args.gt_dir, stem + ".png")
        if not os.path.exists(gp):
            continue

        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        gt = load_mask(gp)
        bbox = mask_to_bbox(gt)

        pred = predictor.predict(img, gt_bbox_xyxy=bbox)

        m_iou = iou(pred, gt)
        prec, rec = precision_recall(pred, gt)
        rows.append({"name": stem, "iou": m_iou, "precision": prec, "recall": rec})

        if viz_saved < args.num_viz:
            out = overlay(img, gt, pred)
            cv2.imwrite(os.path.join(viz_dir, f"{stem}_gt_vs_pred.png"), out)
            viz_saved += 1

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("没有可评测样本：检查mask是否同名png，且路径正确。")

    df.to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)

    summary = {
        "num_samples": int(len(df)),
        "mean_iou": float(df["iou"].mean()),
        "mean_precision": float(df["precision"].mean()),
        "mean_recall": float(df["recall"].mean()),
        "median_iou": float(df["iou"].median()),
    }
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    top = df.sort_values("iou", ascending=False).head(20)
    plt.figure()
    plt.bar(top["name"], top["iou"])
    plt.xticks(rotation=70, fontsize=8)
    plt.ylabel("IoU")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "iou_bar.png"), dpi=200)

    print("Done:", summary)
    print("viz:", viz_dir)


if __name__ == "__main__":
    main()
# python eval_1to3.py --images ./out_eval/vis_gt_egohands --masks ./out_eval/gt_egohands --baseline_mode bbox --num_viz 5 --out_dir ./out_eval/eval_1to3_bbox
