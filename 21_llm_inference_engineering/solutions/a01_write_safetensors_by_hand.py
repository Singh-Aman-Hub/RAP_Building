"""Exercise A1 solution: write a .safetensors file WITHOUT the safetensors library, then read it back with the library.
Layout: 8 bytes little-endian header length N | N bytes JSON header | raw tensor bytes (offsets relative to the data start)."""
import json, os, struct, tempfile
import numpy as np
from safetensors.numpy import load_file

DT = {np.dtype("float32"): "F32", np.dtype("float16"): "F16", np.dtype("int64"): "I64"}

def write_st(path, tensors):
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": DT[arr.dtype], "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw); off += len(raw)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)                       # pad the JSON so the data starts 8-byte aligned
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h))); f.write(h)
        for b in blobs: f.write(b)

rng = np.random.default_rng(0)
tensors = {"a": rng.standard_normal((3, 5)).astype(np.float32),
           "b": rng.standard_normal((4,)).astype(np.float16),
           "c": np.arange(6, dtype=np.int64).reshape(2, 3)}
path = os.path.join(tempfile.mkdtemp(), "mine.safetensors")
write_st(path, tensors)
back = load_file(path)                              # the official reader accepts our file
for k, v in tensors.items():
    assert back[k].dtype == v.dtype and np.array_equal(back[k], v), k
n = struct.unpack("<Q", open(path, "rb").read(8))[0]
print(f"file {os.path.getsize(path)} bytes = 8 + {n} (JSON) + {os.path.getsize(path) - 8 - n} (data)")
print("official safetensors reader round-trips all", len(tensors), "tensors: OK")
