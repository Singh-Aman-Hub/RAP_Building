"""Exercise A4 solution: bytes per block for Q8_0 / Q4_0, bits per weight, file size of a 7B and 70B model, cross-checked with step 02's quantizer."""
import importlib.util, os
import numpy as np

BLOCK = 32
FORMATS = {"Q8_0": 2 + 32 * 1, "Q4_0": 2 + 16, "F16": 32 * 2, "F32": 32 * 4}        # fp16 scale + payload per 32 weights
for name, nbytes in FORMATS.items():
    bpw = nbytes * 8 / BLOCK
    print(f"{name:5s} {nbytes:3d} bytes per 32 weights = {bpw:5.2f} bpw | 7B {7e9*bpw/8/1e9:5.1f} GB | 70B {70e9*bpw/8/1e9:6.1f} GB")
assert FORMATS["Q8_0"] * 8 / BLOCK == 8.5 and FORMATS["Q4_0"] * 8 / BLOCK == 4.5

# cross-check against the real implementation from step 02 (its main() is guarded, import is side-effect free)
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("gg", os.path.join(here, "..", "02_gguf_and_llamacpp_formats", "main.py"))
gg = importlib.util.module_from_spec(spec); spec.loader.exec_module(gg)
x = np.random.default_rng(0).standard_normal(64 * 64).astype(np.float32)
assert len(gg.quantize_q8_0(x)) == FORMATS["Q8_0"] * (x.size // BLOCK)
assert len(gg.quantize_q4_0(x)) == FORMATS["Q4_0"] * (x.size // BLOCK)
print("step 02 quantizers produce exactly 34 and 18 bytes per block: OK")
