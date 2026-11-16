"""Exercise Q2 solution: bits per weight with scale (and zero-point) overhead, and the size of a 70B model."""
def bpw(bits, group, scale_bits=16, zero_bits=0):
    return bits + (scale_bits + zero_bits) / group

assert abs(bpw(4, 128) - 4.125) < 1e-12
for name, args in (("int4 g128 fp16 scale", (4, 128)), ("int4 g64 fp16 scale", (4, 64)), ("int4 g128 scale+zero(fp16)", (4, 128, 16, 16)),
                   ("int3 g128", (3, 128)), ("MXFP4 block32 8-bit scale", (4, 32, 8)), ("NVFP4 block16 8-bit scale", (4, 16, 8))):
    b = bpw(*args)
    print(f"{name:<28} {b:.4f} bpw -> 70B params = {70e9 * b / 8 / 1e9:6.1f} GB")
print("OK")
