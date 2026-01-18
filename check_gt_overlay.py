import argparse
from pathlib import Path
import random
import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw


def poly_to_mask(size_hw, poly_xy):
    """poly_xy: [(x,y),...]"""
    H, W = size_hw
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).polygon(poly_xy, outline=1, fill=1)
    return np.array(m, dtype=bool)


def overlay(img_rgb, mask, alpha=0.45):
    img = img_rgb.astype(np.float32)
    color = np.array([255, 0, 0], dtype=np.float32)[None, None, :]
    m3 = np.repeat(mask[:, :, None], 3, axis=2)
    out = np.where(m3, (1 - alpha) * img + alpha * color, img)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        required=True,
        help="egohands root dir (contains metadata.mat and _LABELLED_SAMPLES)",
    )
    ap.add_argument("--n", type=int, default=3, help="how many frames to sample")
    ap.add_argument("--out", default="./gt_vis_out")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    m = sio.loadmat(root / "metadata.mat", squeeze_me=True, struct_as_record=False)
    videos = m["video"]
    if isinstance(videos, np.ndarray):
        videos = list(videos.flat)

    # 收集所有 labelled frames 的引用
    frame_refs = []
    for vid in videos:
        lf = getattr(vid, "labelled_frames", None)
        if lf is None:
            continue
        if isinstance(lf, np.ndarray):
            lf_list = list(lf.flat)
        elif isinstance(lf, (list, tuple)):
            lf_list = list(lf)
        else:
            lf_list = [lf]
        for fr in lf_list:
            frame_refs.append((vid, fr))

    print("Total labelled frames:", len(frame_refs))
    picks = random.sample(frame_refs, k=min(args.n, len(frame_refs)))

    for i, (vid, fr) in enumerate(picks):
        # TODO: 下面这三行需要用你 deep inspect 得到的真实字段名替换
        # 1) frame path
        # frame_path = root / "_LABELLED_SAMPLES" / <...>
        # 2) polygons for hands (maybe left/right, maybe multiple hands)
        # polys = fr.<something>

        raise RuntimeError(
            "Need labelled_frame field names. Run inspect_meta_deep.py and paste output."
        )


if __name__ == "__main__":
    main()
