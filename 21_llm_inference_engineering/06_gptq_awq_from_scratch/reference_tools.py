"""
REFERENCE - not run in this course (needs CUDA and/or packages that are NOT installed here: auto-gptq/gptqmodel, llmcompressor,
bitsandbytes, torchao, hqq).  This file only PRINTS the recipes.  It never imports those packages, never downloads anything,
never loads a model.  API names change often: always check each project's current docs before copying (as of Nov 2026 the
shapes below are what I am confident of; exact argument names may have moved).

RUN:  python reference_tools.py        (prints the recipes + which packages are importable on this machine)
"""
import importlib.util

RECIPES = {
    "llm-compressor (vLLM project; common way to make quantized checkpoints for vLLM)": '''
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier          # also AWQModifier, QuantizationModifier (data-free FP8 / RTN)
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
oneshot(model="Qwen/Qwen2.5-7B-Instruct", dataset="<calibration dataset>", recipe=recipe,
        max_seq_length=2048, num_calibration_samples=512, output_dir="qwen-w4a16")
# internally: hooks every Linear, feeds calibration batches, builds H = 2 X^T X per layer, runs the GPTQ column loop,
# then saves int4 weights + scales in the compressed-tensors format that vLLM loads directly.''',
    "AutoGPTQ / GPTQModel (GPTQ checkpoints; GPTQModel is the actively maintained successor family - check status)": '''
from gptqmodel import GPTQModel, QuantizeConfig
cfg = QuantizeConfig(bits=4, group_size=128)
m = GPTQModel.load("org/model", cfg); m.quantize(calibration_texts); m.save("model-gptq-4bit")
# internally: exactly the algorithm of main.py section 3 (+ lazy block updates, act-order, fused CUDA int4 kernels at inference).''',
    "AutoAWQ (AWQ checkpoints; the standalone repo has been reported as archived/deprecated - CHECK STATUS, llm-compressor's AWQModifier is the maintained route)": '''
# from awq import AutoAWQForCausalLM
# m = AutoAWQForCausalLM.from_pretrained(path); m.quantize(tok, quant_config={"w_bit": 4, "q_group_size": 128, "zero_point": True, "version": "GEMM"})
# internally: per block, search scales s = mean|X|^alpha (+ clipping search), fold 1/s into the previous op, round to int4 groups.''',
    "bitsandbytes (Hugging Face integration; no calibration data)": '''
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype="bfloat16")
m = AutoModelForCausalLM.from_pretrained("org/model", quantization_config=q)     # or load_in_8bit=True for LLM.int8()
# internally: weights stored as NF4 blocks of 64 (absmax) [double-quantized scales]; dequantized to bf16 on the fly inside each matmul. Used by QLoRA.''',
    "torchao (PyTorch native quantization / sparsity)": '''
from torchao.quantization import quantize_
# quantize_(model, <a weight-only or dynamic-activation config, e.g. int4 weight-only / int8 dynamic>)   <- config class names have changed between releases: check docs
# internally: swaps nn.Linear weights for tensor subclasses holding packed ints + scales; kernels dispatch through torch.compile / custom ops.''',
    "HQQ (Half-Quadratic Quantization; calibration-free)": '''
# from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
# internally: optimises the zero-point (and scale) of each group to minimise a robust |W - Wq| loss with a half-quadratic solver;
# no calibration data, quantizes a 70B model in minutes. Quality is between RTN and GPTQ/AWQ.''',
    "vLLM serving a quantized checkpoint (any of the above)": '''
# vllm serve <quantized-model-dir>          (quantization method is read from the checkpoint config; --quantization is rarely needed)
# on-the-fly FP8:  vllm serve <model> --quantization fp8     |  KV cache:  --kv-cache-dtype fp8''',
}

if __name__ == "__main__":
    print(__doc__)
    for title, body in RECIPES.items():
        print("=" * 100)
        print("REFERENCE:", title)
        print(body)
    print("=" * 100)
    for pkg in ("llmcompressor", "gptqmodel", "auto_gptq", "awq", "bitsandbytes", "torchao", "hqq", "vllm"):
        print(f"  importable here: {pkg:<14} {importlib.util.find_spec(pkg) is not None}")
