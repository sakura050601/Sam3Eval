import scipy.io as sio
import numpy as np

m = sio.loadmat("egohands/metadata.mat", squeeze_me=True, struct_as_record=False)
video = m["video"]  # likely shape (48,) after squeeze
print("video type:", type(video), "len:", len(video))

v0 = video[0]
print("video fields:", [f for f in dir(v0) if not f.startswith("_")])

print("\nExample video_id:", getattr(v0, "video_id", None))
lf = getattr(v0, "labelled_frames", None)
print("labelled_frames type:", type(lf))


def describe(x, name):
    if x is None:
        print(name, "is None")
        return
    if isinstance(x, np.ndarray):
        print(name, "ndarray shape:", x.shape, "dtype:", x.dtype)
        if x.dtype == object and x.size > 0:
            print(name, "first elem type:", type(x.flat[0]))
    else:
        print(name, "type:", type(x))


describe(lf, "labelled_frames")


# 取第一帧看看有哪些字段
# lf 可能是 array/list of structs
def first_elem(x):
    if isinstance(x, np.ndarray):
        return x.flat[0]
    if isinstance(x, (list, tuple)):
        return x[0]
    return x


f0 = first_elem(lf)
print("\nFirst labelled_frame:", type(f0))
print("frame fields:", [f for f in dir(f0) if not f.startswith("_")])

# 把每个字段的类型/shape打出来
for key in [f for f in dir(f0) if not f.startswith("_")]:
    val = getattr(f0, key)
    if isinstance(val, np.ndarray):
        print(f"{key}: ndarray shape={val.shape}, dtype={val.dtype}")
    else:
        print(f"{key}: type={type(val)} value={str(val)[:80]}")
