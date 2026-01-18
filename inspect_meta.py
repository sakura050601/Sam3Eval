#
import scipy.io as sio

m = sio.loadmat("egohands/metadata.mat")
print([k for k in m.keys() if not k.startswith("__")])
for k in [x for x in m.keys() if not x.startswith("__")]:
    v = m[k]
    print(k, type(v), getattr(v, "dtype", None), getattr(v, "shape", None))
