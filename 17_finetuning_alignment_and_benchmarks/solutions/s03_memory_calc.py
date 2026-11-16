"""Exercise 03 (Modify): memory calculator with your own trainable fraction and bytes per parameter.
RUN: .venv/bin/python 17_finetuning_alignment_and_benchmarks/solutions/s03_memory_calc.py"""
def mem(n_params, mode, frac=0.01):
    gb = 1e9
    if mode == "full":   return 16 * n_params / gb                       # bf16 w + bf16 g + fp32 master + Adam m,v
    if mode == "lora":   return (2 * n_params + 16 * frac * n_params) / gb
    if mode == "qlora":  return (0.56 * n_params + 16 * frac * n_params) / gb   # 4-bit NF4 incl. quantisation constants (approx)
    if mode == "full8bit_adam": return (2 + 2 + 4 + 2) * n_params / gb   # 8-bit optimiser states: m,v at 1 byte each
for n in (0.5e9, 7e9, 70e9):
    print(f"{n/1e9:>5.1f}B  full {mem(n,'full'):8.1f}  full+8bitAdam {mem(n,'full8bit_adam'):8.1f}  lora {mem(n,'lora'):7.1f}  qlora {mem(n,'qlora'):7.1f}  (GB, no activations)")
print("A 7B LoRA with r=16 on all linear layers trains roughly 0.5-1% of parameters: try frac=0.005 and frac=0.02.")
