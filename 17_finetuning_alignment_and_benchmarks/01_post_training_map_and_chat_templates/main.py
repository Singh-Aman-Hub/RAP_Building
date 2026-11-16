"""
GOAL:  Draw the whole post-training map (pretrain -> SFT -> preference -> RL -> eval), then SEE its first
       practical ingredient: the chat template. Show the exact string and token ids Qwen2.5 builds from a
       conversation, the four dataset formats, which tokens would carry the SFT loss, and why a BASE model
       (gpt2) just continues text instead of answering. (Only Qwen's tokenizer is cached, not its weights.)

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 17_finetuning_alignment_and_benchmarks/01_post_training_map_and_chat_templates/main.py
EXPECTED OUTPUT (shortened):
    stage table (6 rows)
    '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi<|im_end|>...'
    token ids + pieces, loss-mask preview (only assistant tokens marked "TRAIN"), 4 dataset formats
    base model continues the raw text instead of answering
NEEDS: torch, transformers; cached Qwen/Qwen2.5-0.5B tokenizer and gpt2.  ~15 s on CPU.
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
TOK_NAME = "Qwen/Qwen2.5-0.5B"      # only the TOKENIZER of Qwen is cached offline (the weights are not)
BASE_LM = os.environ.get("BASE_LM", "gpt2")   # a real base (not instruct) model that we can run on a CPU

# ## 1. The post-training map as data
STAGES = [
    ("1 pretraining",      "next-token loss on trillions of web/code tokens", "base model",              "knowledge, language"),
    ("2 mid-training",     "same loss, curated/synthetic/long-context data",  "stronger base",           "reasoning, long context"),
    ("3 SFT",              "cross-entropy on (prompt -> good answer)",        "instruct (chat) model",   "format, following instructions"),
    ("4 preference tuning","DPO-style loss on (chosen, rejected) pairs",      "aligned model",           "tone, helpfulness, safety"),
    ("5 RL (RLHF/RLVR)",   "policy gradient vs reward model or verifier",     "reasoning/agent model",   "verifiable skills (math, code)"),
    ("6 evaluation",       "benchmarks + judges + human arenas",              "a scorecard",             "decide ship / iterate"),
]
print(f"{'stage':<20}{'signal / loss':<52}{'output':<24}teaches")
for s in STAGES:
    print(f"{s[0]:<20}{s[1]:<52}{s[2]:<24}{s[3]}")

# ## 2. Load the tokenizer and build a chat prompt
tok = AutoTokenizer.from_pretrained(TOK_NAME)
messages = [{"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris."}]
text = tok.apply_chat_template(messages, tokenize=False)        # a plain string, no ids yet
print("\n--- chat template output (repr) ---")
print(repr(text))

# ## 3. Generation prompt: where the model should start writing
prompt = tok.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
print("\n--- same, user turn only + add_generation_prompt=True ---")
print(repr(prompt))

# ## 4. Exact token ids and pieces
ids = tok(text, add_special_tokens=False).input_ids              # list[int]; the template already holds the specials
print(f"\nn_tokens={len(ids)}  eos_token={tok.eos_token!r}  pad_token={tok.pad_token!r}  vocab={len(tok)}")
print("id -> piece:")
print("  " + "  ".join(f"{i}:{tok.decode([i])!r}" for i in ids[:14]) + " ...")

# ## 5. Which tokens carry the SFT loss (completion-only mask)
prompt_ids = tok(prompt, add_special_tokens=False).input_ids     # everything the user side produced
full_ids = ids
# The assistant turn = tokens after the generation prompt. Mask prompt positions with -100 (ignored by cross entropy).
labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
print(f"\nprompt tokens={len(prompt_ids)}  trained tokens={sum(l != -100 for l in labels)}")
print("TRAIN tokens:", [tok.decode([l]) for l in labels if l != -100])

# ## 6. The four dataset formats
FORMATS = {
    "prompt/completion":  {"prompt": "Q: 2+2?\nA:", "completion": " 4"},
    "messages (chat)":    {"messages": messages},
    "preference pair":    {"prompt": "Q: 2+2?\nA:", "chosen": " 4", "rejected": " 5"},
    "verifiable (RLVR)":  {"prompt": "Q: 2+2?\nA:", "ground_truth": "4"},
}
print("\n--- dataset formats ---")
for name, row in FORMATS.items():
    print(f"{name:<20}{row}")

# ## 7. A base model does not "answer": it continues text (gpt2 is a base model)
gtok = AutoTokenizer.from_pretrained(BASE_LM)
model = AutoModelForCausalLM.from_pretrained(BASE_LM).eval()
def continue_text(s, n=24):
    enc = gtok(s, return_tensors="pt")                                   # (1, L)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=gtok.eos_token_id)  # greedy
    return gtok.decode(out[0, enc.input_ids.shape[1]:])                  # only the new tokens
print(f"\n--- {BASE_LM} (base model) greedy continuation ---")
print("raw question  :", repr(continue_text("What is the capital of France?")))
print("chat-formatted:", repr(continue_text(prompt)))

# TRY IT: pass a system message ({"role": "system", "content": "Answer in French."}) and print the template again.
# BREAK IT: tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) on the FULL conversation ends with an
#           empty assistant header, so a model trained on it would learn to answer after an answer. Print it and look.
