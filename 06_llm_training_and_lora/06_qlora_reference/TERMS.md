# Terms — QLoRA (Reference)

Folder: `06_llm_training_and_fast_inference/06_qlora_reference` · Files covered: `qlora_reference.py`, `WALKTHROUGH.md` (reference only, not executed; needs an NVIDIA GPU)
Back to module: [06 README](../README.md)

**Terms in this folder:** [QLoRA](#qlora) · [bitsandbytes](#bitsandbytes) · [Quantization (4-bit)](#quantization-4-bit) · [BitsAndBytesConfig](#bitsandbytesconfig) · [load_in_4bit](#load_in_4bit) · [NF4 (NormalFloat4)](#nf4-normalfloat4) · [Double quantization](#double-quantization) · [bnb_4bit_compute_dtype and bfloat16](#bnb_4bit_compute_dtype-and-bfloat16) · [quantization_config](#quantization_config) · [device_map="auto"](#device_mapauto) · [prepare_model_for_kbit_training](#prepare_model_for_kbit_training) · [Paged optimizer (paged_adamw_8bit)](#paged-optimizer-paged_adamw_8bit) · [De-quantisation on the fly](#de-quantisation-on-the-fly) · [Training memory: full vs LoRA vs QLoRA](#training-memory-full-vs-lora-vs-qlora) · [py_compile](#py_compile)

---

## QLoRA  `(concept)`
**1. Definition:** QLoRA fine-tunes a model with LoRA adapters on top of a base model whose frozen weights are stored in 4-bit precision.
**2. In plain words:** The big frozen part is shrunk to a quarter of its fp16 size, and only the small 16-bit adapters learn. A 7B model's weights go from about 14 GB to about 4 GB, so it fits a single 16 GB GPU. The cost is a little quantisation error and slower steps, and in practice it needs CUDA.
**3. Example to quote:** > "QLoRA is LoRA over a 4-bit base: weights drop from roughly 14 GB to 4 GB, so I can fine-tune a 7B model on one consumer GPU."
**Seen in:** `qlora_reference.py` docstring; `WALKTHROUGH.md`

## bitsandbytes  `(third-party: bitsandbytes)`
**1. Definition:** `bitsandbytes` is a library providing low-precision (8-bit and 4-bit) CUDA kernels and optimizers used by Transformers for quantised loading and training.
**2. In plain words:** It's the engine that makes 4-bit weights usable on a GPU. It does not run on Apple-silicon Macs, which is why this file is read-only here.
**3. Example to quote:** > "QLoRA relies on `bitsandbytes`, which needs an NVIDIA GPU, so I'd run it in a Colab notebook, not on my Mac."
**Seen in:** `qlora_reference.py`, docstring and imports

## Quantization (4-bit)  `(concept)`
**1. Definition:** Quantization stores numbers with fewer bits (here 4 instead of 16) by mapping them to a small set of discrete levels plus scale factors.
**2. In plain words:** It trades a bit of accuracy for large memory savings. Four bits means 16 possible levels per weight.
**3. Example to quote:** > "Storing weights in 4 bits uses 0.5 bytes per parameter instead of 2 for fp16."
**Seen in:** `qlora_reference.py`

## BitsAndBytesConfig  `(third-party: transformers)`
**1. Definition:** `BitsAndBytesConfig(...)` is a settings object telling `from_pretrained` how to quantise weights while loading.
**2. In plain words:** It's the recipe for the 4-bit loading: type, double quantization and compute dtype.
**3. Example to quote:** > "I build `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', ...)` and pass it when loading the model."
**Seen in:** `qlora_reference.py`

## load_in_4bit  `(concept)`
**1. Definition:** `load_in_4bit=True` makes the model's frozen weights be stored as 4-bit numbers at load time.
**2. In plain words:** It's the switch that shrinks memory use. 
**3. Example to quote:** > "`load_in_4bit=True` is what turns plain LoRA into QLoRA."
**Seen in:** `qlora_reference.py`

## NF4 (NormalFloat4)  `(concept)`
**1. Definition:** NF4 is a 4-bit number format whose 16 levels are placed to match the bell-shaped (normal) distribution of weight values.
**2. In plain words:** Weights cluster near zero, so putting more levels there loses less accuracy than evenly spaced int4.
**3. Example to quote:** > "I choose `bnb_4bit_quant_type='nf4'` because its levels fit how weights are actually distributed."
**Seen in:** `qlora_reference.py`; `WALKTHROUGH.md`

## Double quantization  `(concept)`
**1. Definition:** Double quantization also quantises the per-block scaling constants used by the 4-bit scheme.
**2. In plain words:** It saves roughly 0.4 bit per parameter at no practical quality cost.
**3. Example to quote:** > "`bnb_4bit_use_double_quant=True` quantises the scale factors too, saving about 0.4 bits per parameter."
**Seen in:** `qlora_reference.py`

## bnb_4bit_compute_dtype and bfloat16  `(concept)`
**1. Definition:** `bnb_4bit_compute_dtype` is the precision (here `torch.bfloat16`) in which the matrix multiplications run after a weight block is de-quantised; bfloat16 is a 16-bit float format with the exponent range of fp32.
**2. In plain words:** Storage is 4-bit, but the arithmetic needs more precision to stay stable, so blocks are expanded to bf16 for the multiply.
**3. Example to quote:** > "Weights are stored in 4-bit but computed in bf16, so I keep stability without storing 16-bit weights."
**Seen in:** `qlora_reference.py`

## quantization_config  `(third-party: transformers)`
**1. Definition:** The `quantization_config` argument of `from_pretrained` passes the quantisation settings to the model loader.
**2. In plain words:** It's the only line that makes the loading quantised.
**3. Example to quote:** > "`from_pretrained(MODEL_NAME, quantization_config=bnb_config)` loads the weights already in 4-bit."
**Seen in:** `qlora_reference.py`

## device_map="auto"  `(third-party: accelerate)`
**1. Definition:** `device_map="auto"` lets the accelerate library decide which layers go on which available devices (GPUs, CPU).
**2. In plain words:** It spreads a large model over what hardware you have without you doing it by hand.
**3. Example to quote:** > "I use `device_map='auto'` so layers are placed on the available GPU automatically."
**Seen in:** `qlora_reference.py`

## prepare_model_for_kbit_training  `(third-party: peft)`
**1. Definition:** A peft helper that readies a quantised model for training: casts layer norms to fp32, enables gradient-checkpointing hooks and makes input embeddings require gradients.
**2. In plain words:** A 4-bit model needs a few tweaks before gradients can flow properly through the frozen base to the adapters.
**3. Example to quote:** > "I call `prepare_model_for_kbit_training(model)` before attaching LoRA to the 4-bit model."
**Seen in:** `qlora_reference.py`

## Paged optimizer (paged_adamw_8bit)  `(concept)`
**1. Definition:** A paged optimizer keeps optimizer states in memory that can spill to CPU RAM when the GPU temporarily runs out, and `PagedAdamW8bit` also stores them in 8-bit.
**2. In plain words:** It prevents out-of-memory crashes during memory spikes.
**3. Example to quote:** > "I'd use `PagedAdamW8bit` so the optimizer state pages to CPU RAM instead of crashing the GPU."
**Seen in:** `qlora_reference.py`, step 4 comments

## De-quantisation on the fly  `(concept)`
**1. Definition:** During the forward pass, each 4-bit block is converted to bf16, used in the matmul, and discarded.
**2. In plain words:** That's why QLoRA is slower per step than LoRA: extra conversion work. Only adapters receive gradients.
**3. Example to quote:** > "The frozen 4-bit weights are expanded block by block just in time, which costs speed but saves memory."
**Seen in:** `WALKTHROUGH.md`

## Training memory: full vs LoRA vs QLoRA  `(concept)`
**1. Definition:** The memory needed for weights, gradients and optimizer states differs by method.
**2. In plain words:** For a 6.7B model, the memory calculator reports 107.2 GB for full fine-tuning, 13.5 GB for LoRA and 3.5 GB for QLoRA, before activations.
**3. Example to quote:** > "Full fine-tuning of a 6.7B model needs about 107 GB, LoRA 13.5 GB and QLoRA 3.5 GB."
**Seen in:** `WALKTHROUGH.md`

## py_compile  `(command)`
**1. Definition:** `python -m py_compile file.py` compiles a Python file to bytecode, which checks syntax without running it.
**2. In plain words:** It's how this reference file is verified on a machine that can't run it.
**3. Example to quote:** > "I ran `python -m py_compile qlora_reference.py` to check the syntax without needing a GPU."
```bash
python -m py_compile qlora_reference.py
```
**Seen in:** `WALKTHROUGH.md`
