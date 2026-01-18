import argparse
from pathlib import Path
import random
import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw

CANDIDATES = [
    "{video}/{frame}.jpg",
    "{video}/{frame}.png",
    "{video}/frame_{frame}.jpg",
    "{video}/frame_{frame}.png",
    "{video}/{frame:06d}.jpg",
    "{video}/{frame:06d}.png",
    "{video}/frame_{frame:06d}.jpg",
    "{video}/frame_{frame:06d}.png",
]


def polygon_to_mask(hw, poly_xy):
    """hw=(H,W), poly_xy ndarray (N,2) float -> bool mask"""
    if poly_xy is None or len(poly_xy) == 0:
        return None
    H, W = hw
    m = Image.new("L", (W, H), 0)
    pts = [(float(x), float(y)) for x, y in poly_xy]
    ImageDraw.Draw(m).polygon(pts, outline=1, fill=1)
    return np.array(m, dtype=bool)


def overlay(img_u8, mask, alpha=0.45):
    img = img_u8.astype(np.float32)
    color = np.array([255, 0, 0], dtype=np.float32)[None, None, :]
    m3 = np.repeat(mask[:, :, None], 3, axis=2)
    out = np.where(m3, (1 - alpha) * img + alpha * color, img)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def find_frame_image(root: Path, video_id: str, frame_num: int) -> Path | None:
    base = root / "_LABELLED_SAMPLES"
    for pat in CANDIDATES:
        rel = pat.format(video=video_id, frame=frame_num)
        p = base / rel
        if p.exists():
            return p
    vdir = base / video_id
    if vdir.exists():
        for p in vdir.glob("*"):
            if (
                p.suffix.lower() in [".jpg", ".jpeg", ".png"]
                and str(frame_num) in p.stem
            ):
                return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--egohands_root",
        required=True,
        help="folder that contains metadata.mat and _LABELLED_SAMPLES/",
    )
    ap.add_argument(
        "--n", type=int, default=3, help="how many labelled frames to visualize"
    )
    ap.add_argument("--out", default="./gt_vis_out", help="output folder")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    root = Path(args.egohands_root)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    m = sio.loadmat(root / "metadata.mat", squeeze_me=True, struct_as_record=False)
    videos = (
        list(m["video"].flat) if isinstance(m["video"], np.ndarray) else [m["video"]]
    )

    # 收集所有 (video_id, frame_struct)
    refs = []
    for vid in videos:
        video_id = str(getattr(vid, "video_id"))
        lf = getattr(vid, "labelled_frames")
        lf_list = list(lf.flat) if isinstance(lf, np.ndarray) else [lf]
        for fr in lf_list:
            refs.append((video_id, fr))

    print("Total labelled frames:", len(refs))
    picks = random.sample(refs, k=min(args.n, len(refs)))

    for idx, (video_id, fr) in enumerate(picks):
        frame_num = int(getattr(fr, "frame_num"))
        img_path = find_frame_image(root, video_id, frame_num)
        if img_path is None:
            print(f"[{idx}] NOT FOUND image for {video_id} frame={frame_num}")
            continue

        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        H, W = img.shape[:2]

        # 4个手的polygon
        polys = {
            "myleft": getattr(fr, "myleft"),
            "myright": getattr(fr, "myright"),
            "yourleft": getattr(fr, "yourleft"),
            "yourright": getattr(fr, "yourright"),
        }

        # 生成 union GT mask
        union = np.zeros((H, W), dtype=bool)
        for name, poly in polys.items():
            if isinstance(poly, np.ndarray) and poly.size > 0:
                msk = polygon_to_mask((H, W), poly)
                if msk is not None:
                    union |= msk

        # 保存
        stem = f"{video_id}_f{frame_num:06d}"
        Image.fromarray((union.astype(np.uint8) * 255), mode="L").save(
            outdir / f"{stem}_gt.png"
        )
        overlay(img, union).save(outdir / f"{stem}_overlay.png")
        # 也把原图拷贝一份方便对照
        Image.fromarray(img).save(outdir / f"{stem}_img.png")

        # 打印基本自检信息
        area = union.mean()
        print(f"[{idx}] {stem} -> {img_path.name}  hand_area_ratio={area:.4f}")

    print("Saved to:", outdir.resolve())


if __name__ == "__main__":
    main()
# python vis_gt_egohands.py --egohands_root egohands --n 3
