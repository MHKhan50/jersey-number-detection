"""Export a JerseyNet checkpoint to ONNX for the browser demo, without needing PyTorch.

    python demo/export_onnx.py best.pt demo/jerseynet.onnx

Batch norm is folded into the convolutions. The script then runs the ONNX model with ONNX
Runtime and compares it with a NumPy re-implementation of the network on a random input.
"""

import pickle
import struct
import sys
import zipfile

import numpy as np
import onnxruntime as ort

CKPT, OUT = sys.argv[1], sys.argv[2]

# ---------------------------------------------------------------- load checkpoint
zf = zipfile.ZipFile(CKPT)
names = zf.namelist()
prefix = names[0].split("/")[0]
DT = {"FloatStorage": np.float32, "LongStorage": np.int64}


class Storage:
    def __init__(self, kind):
        self.kind = kind


class U(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod == "torch._utils" and name == "_rebuild_tensor_v2":

            def rebuild(st, off, size, stride, *a):
                arr = st
                if not size:
                    return arr[off]
                return np.lib.stride_tricks.as_strided(
                    arr[off:], shape=size, strides=[s * arr.itemsize for s in stride]
                ).copy()

            return rebuild
        if mod == "torch" and name in DT:
            return name
        if mod == "collections" and name == "OrderedDict":
            import collections

            return collections.OrderedDict
        if mod == "torch._utils" and name == "_rebuild_parameter":
            return lambda d, *a: d
        return super().find_class(mod, name)

    def persistent_load(self, pid):
        _, kind, key, loc, numel = pid
        raw = zf.read(f"{prefix}/data/{key}")
        return np.frombuffer(raw, dtype=DT[kind]).copy()


state = U(zf.open(f"{prefix}/data.pkl")).load()
sd = state["model"]
cfg = state.get("config", {})
print("run:", cfg.get("out_dir"), "epoch", state.get("epoch"), "val AP", round(state["val"]["ap50_95"], 4))


# ---------------------------------------------------------------- fold batch norm
def fold(conv, bn, eps=1e-5):
    w = sd[conv + ".weight"]
    b = sd.get(conv + ".bias", np.zeros(w.shape[0], np.float32))
    g, beta = sd[bn + ".weight"], sd[bn + ".bias"]
    mu, var = sd[bn + ".running_mean"], sd[bn + ".running_var"]
    s = g / np.sqrt(var + eps)
    return (w * s[:, None, None, None]).astype(np.float32), ((b - mu) * s + beta).astype(np.float32)


# ---------------------------------------------------------------- protobuf encoder
def varint(n):
    out = b""
    n &= (1 << 64) - 1
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out += bytes([b | 0x80])
        else:
            return out + bytes([b])


def key(f, wt):
    return varint((f << 3) | wt)


def fint(f, v):
    return key(f, 0) + varint(v)


def fbytes(f, b):
    return key(f, 2) + varint(len(b)) + b


def fstr(f, s):
    return fbytes(f, s.encode())


def ffloat(f, v):
    return key(f, 5) + struct.pack("<f", v)


def tensor(name, arr):
    arr = np.ascontiguousarray(arr)
    dt = 1 if arr.dtype == np.float32 else 7
    return b"".join(fint(1, d) for d in arr.shape) + fint(2, dt) + fstr(8, name) + fbytes(9, arr.tobytes())


def attr_i(n, v):
    return fstr(1, n) + fint(20, 2) + fint(3, v)


def attr_s(n, v):
    return fstr(1, n) + fint(20, 3) + fbytes(4, v.encode())


def attr_ints(n, vs):
    return fstr(1, n) + fint(20, 7) + b"".join(fint(8, v) for v in vs)


def node(op, ins, outs, name, attrs=()):
    return (
        b"".join(fstr(1, i) for i in ins)
        + b"".join(fstr(2, o) for o in outs)
        + fstr(3, name)
        + fstr(4, op)
        + b"".join(fbytes(5, a) for a in attrs)
    )


def vinfo(name, dims):
    shape = b"".join(fbytes(1, fint(1, d) if isinstance(d, int) else fstr(2, d)) for d in dims)
    return fstr(1, name) + fbytes(2, fbytes(1, fint(1, 1) + fbytes(2, shape)))


nodes, inits = [], []
cnt = [0]


def uid(p):
    cnt[0] += 1
    return f"{p}_{cnt[0]}"


def conv(x, w, b, stride=1, relu=True):
    wn, bn_ = uid("W"), uid("B")
    inits.append(tensor(wn, w))
    inits.append(tensor(bn_, b))
    k = w.shape[-1]
    p = k // 2
    y = uid("conv")
    nodes.append(
        node(
            "Conv",
            [x, wn, bn_],
            [y],
            uid("Conv"),
            [
                attr_ints("kernel_shape", [k, k]),
                attr_ints("strides", [stride, stride]),
                attr_ints("pads", [p, p, p, p]),
            ],
        )
    )
    if relu:
        r = uid("relu")
        nodes.append(node("Relu", [y], [r], uid("Relu")))
        return r
    return y


def add(a, b):
    y = uid("add")
    nodes.append(node("Add", [a, b], [y], uid("Add")))
    return y


def relu(a):
    y = uid("relu")
    nodes.append(node("Relu", [a], [y], uid("Relu")))
    return y


def convbn(x, prefix, stride=1, relu_=True):
    w, b = fold(prefix + ".0", prefix + ".1")
    return conv(x, w, b, stride, relu_)


# ---------------------------------------------------------------- graph
x = "image"
x = convbn(x, "stem.0", 2)
x = convbn(x, "stem.1", 1)
feats = []
for s in range(4):
    for blk in range(2):
        p = f"stages.{s}.{blk}"
        stride = 2 if blk == 0 else 1
        h = convbn(x, p + ".conv1", stride)
        h = convbn(h, p + ".conv2", 1, relu_=False)
        idn = convbn(x, p + ".down", stride, relu_=False) if f"{p}.down.0.weight" in sd else x
        x = relu(add(h, idn))
    feats.append(x)
scales = uid("scales")
inits.append(tensor(scales, np.array([1, 1, 2, 2], np.float32)))
pf = conv(feats[3], sd["lateral.3.weight"], sd["lateral.3.bias"], relu=False)
for i in (2, 1, 0):
    lat = conv(feats[i], sd[f"lateral.{i}.weight"], sd[f"lateral.{i}.bias"], relu=False)
    up = uid("up")
    nodes.append(
        node(
            "Resize",
            [pf, "", scales],
            [up],
            uid("Resize"),
            [
                attr_s("mode", "nearest"),
                attr_s("coordinate_transformation_mode", "asymmetric"),
                attr_s("nearest_mode", "floor"),
            ],
        )
    )
    pf = convbn(add(lat, up), f"smooth.{i}")
outs = {}
for head, oname in (("heatmap", "heatmap"), ("wh", "wh"), ("offset", "offset")):
    h = convbn(pf, f"{head}.0")
    y = conv(h, sd[f"{head}.1.weight"], sd[f"{head}.1.bias"], relu=False)
    nodes.append(node("Identity", [y], [oname], uid("Id")))
S = cfg.get("image_size", 384)
graph = (
    b"".join(fbytes(1, n) for n in nodes)
    + fstr(2, "jerseynet")
    + b"".join(fbytes(5, t) for t in inits)
    + fbytes(11, vinfo("image", [1, 3, S, S]))
    + b"".join(
        fbytes(12, vinfo(o, [1, c, S // 4, S // 4])) for o, c in (("heatmap", 1), ("wh", 2), ("offset", 2))
    )
)
model = (
    fint(1, 7)
    + fstr(2, "jersey-number-detection")
    + fstr(3, "2.1.0")
    + fbytes(7, graph)
    + fbytes(8, fint(2, 13))
)
open(OUT, "wb").write(model)
print("wrote", OUT, len(model) // 1024, "KB")


# ---------------------------------------------------------------- numpy reference (no folding)
def np_conv(x, w, b, stride):
    k = w.shape[-1]
    p = k // 2
    x = np.pad(x, ((0, 0), (p, p), (p, p)))
    C, H, W = x.shape
    oh, ow = (H - k) // stride + 1, (W - k) // stride + 1
    cols = np.lib.stride_tricks.as_strided(
        x,
        (C, k, k, oh, ow),
        (x.strides[0], x.strides[1], x.strides[2], x.strides[1] * stride, x.strides[2] * stride),
    )
    y = np.tensordot(w, cols, axes=([1, 2, 3], [0, 1, 2]))
    return y + (b[:, None, None] if b is not None else 0)


def np_bn(x, p, eps=1e-5):
    return (x - sd[p + ".running_mean"][:, None, None]) / np.sqrt(
        sd[p + ".running_var"][:, None, None] + eps
    ) * sd[p + ".weight"][:, None, None] + sd[p + ".bias"][:, None, None]


def np_cbr(x, p, stride=1, r=True):
    y = np_bn(np_conv(x, sd[p + ".0.weight"], None, stride), p + ".1")
    return np.maximum(y, 0) if r else y


def np_forward(img):
    x = np_cbr(img, "stem.0", 2)
    x = np_cbr(x, "stem.1")
    fs = []
    for s in range(4):
        for blk in range(2):
            p = f"stages.{s}.{blk}"
            st = 2 if blk == 0 else 1
            h = np_cbr(np_cbr(x, p + ".conv1", st), p + ".conv2", 1, False)
            idn = np_cbr(x, p + ".down", st, False) if f"{p}.down.0.weight" in sd else x
            x = np.maximum(h + idn, 0)
        fs.append(x)
    q = np_conv(fs[3], sd["lateral.3.weight"], sd["lateral.3.bias"], 1)
    for i in (2, 1, 0):
        lat = np_conv(fs[i], sd[f"lateral.{i}.weight"], sd[f"lateral.{i}.bias"], 1)
        q = np_cbr(lat + q.repeat(2, 1).repeat(2, 2), f"smooth.{i}")
    return {
        h: np_conv(np_cbr(q, f"{h}.0"), sd[f"{h}.1.weight"], sd[f"{h}.1.bias"], 1)
        for h in ("heatmap", "wh", "offset")
    }



sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
rng = np.random.default_rng(0)
img = rng.normal(0, 1, (3, S, S)).astype(np.float32)
o = dict(zip([x.name for x in sess.get_outputs()], sess.run(None, {"image": img[None]}), strict=True))
ref = np_forward(img.astype(np.float64))
for k in ref:
    print(k, "max abs diff", float(np.abs(o[k][0] - ref[k]).max()), "range", float(np.abs(ref[k]).max()))
