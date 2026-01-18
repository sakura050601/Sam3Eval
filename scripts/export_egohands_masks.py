import os, glob, argparse, shutil
import numpy as np
import cv2
from scipy.io import loadmat

FIELDS = ["myleft", "myright", "yourleft", "yourright"]


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def unpack_poly(obj):
    """
    把 mat 里的 object 展开成 (N,2) 或 (2,N) 的 float 数组。
    为空则返回 None。
    """
    if obj is None:
        return None
    a = np.array(obj)
    # 可能是 size=0 的空
    if a.size == 0:
        return None
    # 常见：a 是 object 包一层，比如 array([[array([...])]], dtype=object)
    while isinstance(a, np.ndarray) and a.dtype == object:
        # 找到第一个非空元素继续拆
        flat = [x for x in a.flatten() if x is not None and np.array(x).size > 0]
        if not flat:
            return None
        a = np.array(flat[0])
    # 现在希望是数值数组
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 2:
        return None
    if a.shape[0] == 2 and a.shape[1] != 2:
        a = a.T  # (2,N)->(N,2)
    if a.shape[1] != 2 or a.shape[0] < 3:
        return None
    # 去掉 NaN 行
    a = a[~np.isnan(a).any(axis=1)]
    if len(a) < 3:
        return None
    return a


def to_xy_pixels(poly, H, W):
    """
    把 poly 转成像素坐标 xy：
    - 归一化(0..1) -> 乘以 W/H
    - 1-index -> 0-index（EgoHands 可能是 1-index）
    - 自动判断 xy vs yx
    """
    a = poly.copy().astype(np.float32)

    mn, mx = float(np.min(a)), float(np.max(a))

    # 归一化
    if mx <= 1.5:
        a[:, 0] *= W - 1
        a[:, 1] *= H - 1

    # 1-index（常见情况）：最小值>=1 直接减1
    if mn >= 1.0:
        a -= 1.0

    # 判断 xy/yx：看哪一列更像宽度范围
    c0_min, c0_max = float(a[:, 0].min()), float(a[:, 0].max())
    c1_min, c1_max = float(a[:, 1].min()), float(a[:, 1].max())

    score_xy = int(0 <= c0_min <= W and 0 <= c0_max <= W) + int(
        0 <= c1_min <= H and 0 <= c1_max <= H
    )
    score_yx = int(0 <= c0_min <= H and 0 <= c0_max <= H) + int(
        0 <= c1_min <= W and 0 <= c1_max <= W
    )

    if score_yx > score_xy:
        a = a[:, [1, 0]]  # yx -> xy

    a[:, 0] = np.clip(a[:, 0], 0, W - 1)
    a[:, 1] = np.clip(a[:, 1], 0, H - 1)

    return np.round(a).astype(np.int32)


def polylist_to_mask(poly_list, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)
    for poly in poly_list:
        if poly is None:
            continue
        pts = to_xy_pixels(poly, H, W)
        if pts is None or len(pts) < 3:
            continue
        cv2.fillPoly(mask, [pts], 1)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--video_dir", required=True, help="包含 polygons.mat 和 *.jpg 的那个目录"
    )
    ap.add_argument("--out_images", default="data/images")
    ap.add_argument("--out_masks", default="data/gt_masks")
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--copy_images", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.out_images)
    ensure_dir(args.out_masks)

    mat_path = os.path.join(args.video_dir, "polygons.mat")
    if not os.path.exists(mat_path):
        raise RuntimeError(f"没找到 {mat_path}")

    frames = sorted(glob.glob(os.path.join(args.video_dir, "*.jpg")))
    if not frames:
        raise RuntimeError("video_dir 下没找到 jpg 帧")

    d = loadmat(mat_path)
    polys = d["polygons"]  # shape (1,100), dtype has fields
    if polys.shape[0] != 1:
        raise RuntimeError(f"意外的 polygons shape: {polys.shape}")

    n_anno = polys.shape[1]
    n = min(args.num, n_anno, len(frames))

    saved = 0
    for i in range(n_anno):
        if saved >= args.num:
            break

        # 帧图按排序对齐第 i 张（EgoHands 通常一一对应）
        if i >= len(frames):
            break
        img_path = frames[i]
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        rec = polys[0, i]  # 这一帧的记录
        poly_list = []
        for f in FIELDS:
            try:
                p = unpack_poly(rec[f])
            except Exception:
                p = None
            if p is not None:
                poly_list.append(p)

        mask = polylist_to_mask(poly_list, H, W)

        # 跳过全黑（说明这一帧没标注或对齐失败）
        if mask.sum() == 0:
            continue

        idx = saved + 1
        out_img = os.path.join(args.out_images, f"{idx:04d}.jpg")
        out_msk = os.path.join(args.out_masks, f"{idx:04d}.png")

        if args.copy_images:
            shutil.copy2(img_path, out_img)
        else:
            try:
                if os.path.exists(out_img):
                    os.remove(out_img)
                os.symlink(os.path.abspath(img_path), out_img)
            except Exception:
                shutil.copy2(img_path, out_img)

        cv2.imwrite(out_msk, mask * 255)
        saved += 1

    print(f"exported {saved} pairs")
    print("images:", args.out_images)
    print("masks :", args.out_masks)
    if saved == 0:
        print(
            "还是全黑的话：说明(1)帧顺序和polygons索引不一致，或(2)坐标规则（归一化/xy-yx/1-index）不同，需要进一步打印一帧的多边形数值范围。"
        )


if __name__ == "__main__":
    main()
