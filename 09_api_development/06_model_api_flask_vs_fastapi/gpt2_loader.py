"""
gpt2_loader.py -- load GPT-2 ONCE and generate text from it; built to sit behind an API.

It is the LLM twin of model_loader.py:
    model_loader.py   -> a small scikit-learn classifier  (predict_one)
    gpt2_loader.py    -> a real language model, GPT-2      (generate)
Same ideas: load at start-up (not per request), keep the model in memory, wrap it in one
tidy function the web framework calls, validate input, report timing.

RUN THE SELF-TEST (first run downloads GPT-2, about 500 MB, into ~/.cache/huggingface):
    python gpt2_loader.py                      (from this folder)
    python 09_api_development/06_model_api_flask_vs_fastapi/gpt2_loader.py     (from the repo root)

USE IT FROM FASTAPI (sketch -- the same two lines work in Flask):
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    from gpt2_loader import get_generator

    @asynccontextmanager
    async def lifespan(app):
        get_generator()                  # load GPT-2 at start-up so the first request is not slow
        yield

    app = FastAPI(lifespan=lifespan)

    class GenerateIn(BaseModel):
        prompt: str = Field(min_length=1)
        max_new_tokens: int = Field(default=30, ge=1, le=200)
        temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    @app.post("/generate")
    def generate(body: GenerateIn):      # plain `def`: FastAPI runs it in a thread pool
        try:
            return get_generator().generate(**body.model_dump())
        except ValueError as err:        # e.g. prompt too long for GPT-2's 1024-token window
            raise HTTPException(status_code=422, detail=str(err))

SETTINGS (environment variables, read when the generator is created):
    GPT2_MODEL_NAME   default "gpt2"   (also works: "distilgpt2" smaller/faster, "gpt2-medium" bigger)
    GPT2_DEVICE       default "cpu"    ("cuda" if you have an NVIDIA GPU, "mps" on Apple silicon)

CONCEPTS USED (details: 04_llm_fundamentals, 05_llm_parameters, 06_llm_training_and_lora):
    tokenizer, logits, sampling (temperature / top-k / top-p / repetition penalty),
    context window, KV cache (generate() uses it automatically), inference mode, lazy singleton.
"""

import os
import threading
import time

import torch                                     # 3rd-party: the tensor / neural-network library GPT-2 runs on
from transformers import AutoModelForCausalLM, AutoTokenizer   # 3rd-party: Hugging Face model + tokenizer loaders

DEFAULT_MODEL_NAME = "gpt2"                      # GPT-2 small: 124 million parameters, 1024-token context window


class GPT2Generator:
    """Owns the tokenizer + model. Create ONE per process and reuse it for every request."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = "cpu"):
        self.model_name = model_name
        self.device = device

        # Tokenizer: text <-> token ids. It is tiny and fast to load.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # GPT-2 was trained without a padding token. generate() wants one defined, so we reuse
        # the end-of-text token. (With a single prompt per call nothing is actually padded.)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # The model: ~124M weights. from_pretrained downloads once, then reads from the local cache.
        # "CausalLM" = the head that predicts the NEXT token, which is what text generation needs.
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)                    # move the weights to cpu / cuda / mps
        self.model.eval()                        # inference mode: switches off dropout (randomness used for training)

        # GPT-2's maximum number of tokens (prompt + generated). 1024 for every GPT-2 size.
        self.max_context = self.model.config.n_positions

        # model.generate() on one shared model object is not designed for simultaneous calls from
        # several threads (FastAPI runs plain `def` endpoints in a thread pool), so serialise them.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ info
    def info(self) -> dict:
        """Cheap facts about the loaded model; handy for a /health or /model-info endpoint."""
        n_params = sum(p.numel() for p in self.model.parameters())   # numel() = number of values in a tensor
        return {
            "model_name": self.model_name,
            "device": self.device,
            "parameters": n_params,
            "approx_weights_mb": round(n_params * 4 / 1e6),          # float32 = 4 bytes per parameter
            "max_context_tokens": self.max_context,
        }

    def count_tokens(self, text: str) -> int:
        """How many GPT-2 tokens does `text` use? (Prompt cost and the context limit are counted in tokens.)"""
        return len(self.tokenizer.encode(text))

    # -------------------------------------------------------------- generate
    def generate(self, prompt: str, max_new_tokens: int = 30, temperature: float = 0.0,
                 top_k: int = 50, top_p: float = 1.0, repetition_penalty: float = 1.0,
                 seed: int | None = None) -> dict:
        """Continue `prompt` and return a dict with the NEW text only (not the prompt repeated).

        temperature         0 -> greedy (always the most likely token, deterministic). >0 -> random sampling.
        top_k, top_p        keep only the k most likely tokens / the smallest set with total probability p.
        repetition_penalty  >1.0 discourages repeating tokens already seen (1.0 = off).
        seed                makes sampling repeatable (ignored when temperature == 0).
        Raises ValueError for bad input so the API layer can answer 422 instead of crashing with a 500.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")

        start = time.perf_counter()

        # 1) text -> token ids.  return_tensors="pt" gives PyTorch tensors; shape (1, n_tokens).
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        # 2) Context-window check: prompt + new tokens must fit in 1024, otherwise GPT-2 errors or degrades.
        if prompt_tokens + max_new_tokens > self.max_context:
            raise ValueError(
                f"prompt uses {prompt_tokens} tokens + max_new_tokens {max_new_tokens} "
                f"> context window {self.max_context}"
            )

        # 3) Build the generation settings. Greedy and sampling need different arguments:
        #    passing temperature/top_k/top_p while do_sample=False only triggers warnings.
        do_sample = temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_k=top_k, top_p=top_p)

        # 4) Generate. The lock keeps threads from using the model at the same time.
        with self._lock:
            if do_sample and seed is not None:
                torch.manual_seed(seed)          # global RNG, so it must happen inside the lock too
            with torch.inference_mode():         # no gradient bookkeeping: faster and uses less memory
                output_ids = self.model.generate(**inputs, **gen_kwargs)

        # 5) output_ids = prompt tokens + new tokens. Slice off the prompt, then ids -> text.
        new_ids = output_ids[0][prompt_tokens:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        elapsed = time.perf_counter() - start
        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "new_tokens": int(new_ids.shape[0]),
            "latency_ms": round(elapsed * 1000, 1),
            "tokens_per_second": round(new_ids.shape[0] / elapsed, 1) if elapsed > 0 else None,
            "model_name": self.model_name,
        }


# ---------------------------------------------------------------------------
# Singleton: one generator per process, created the first time it is asked for.
# ---------------------------------------------------------------------------
_GENERATOR: GPT2Generator | None = None
_CREATE_LOCK = threading.Lock()                  # two simultaneous first requests must not load the model twice


def get_generator() -> GPT2Generator:
    """Return the process-wide GPT2Generator, creating (and loading GPT-2) on first call.

    Call it once at start-up (lifespan / before_first_request equivalent) to avoid a slow first request.
    Remember: with Gunicorn/Uvicorn --workers N, EACH worker process loads its own copy (~500 MB for GPT-2).
    """
    global _GENERATOR
    if _GENERATOR is None:
        with _CREATE_LOCK:
            if _GENERATOR is None:               # check again: another thread may have finished while we waited
                _GENERATOR = GPT2Generator(
                    model_name=os.getenv("GPT2_MODEL_NAME", DEFAULT_MODEL_NAME),
                    device=os.getenv("GPT2_DEVICE", "cpu"),
                )
    return _GENERATOR


# ---------------------------------------------------------------------------
# Self-test: python gpt2_loader.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.perf_counter()
    gen = get_generator()
    print(f"Loaded in {time.perf_counter() - t0:.1f}s ->", gen.info())

    prompt = "The best way to learn machine learning is"
    print("\nPrompt tokens:", gen.count_tokens(prompt))

    # TRY IT: change one setting at a time and compare the outputs.
    print("\n[greedy, temperature=0]  (run it twice: identical output)")
    print(gen.generate(prompt, max_new_tokens=25))

    print("\n[sampling, temperature=0.8, seed=42]  (same seed -> same output)")
    print(gen.generate(prompt, max_new_tokens=25, temperature=0.8, seed=42))

    print("\n[sampling, temperature=1.5, seed=42]  (hotter -> wilder text)")
    print(gen.generate(prompt, max_new_tokens=25, temperature=1.5, seed=42))

    # BREAK IT: this 8-token prompt + 1020 new tokens exceeds the 1024-token window -> ValueError (an API would return 422).
    try:
        gen.generate(prompt, max_new_tokens=1020)
    except ValueError as err:
        print("\nExpected error:", err)
