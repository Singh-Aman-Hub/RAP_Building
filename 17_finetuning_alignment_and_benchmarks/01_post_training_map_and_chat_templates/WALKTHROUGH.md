# Step 01 — The post-training map and chat templates

## The story so far
[Module 06](../../06_llm_training_and_lora/README.md) taught you the *mechanics* of training: a loop, a loss, perplexity, checkpoints, and one LoRA run ([06 step 05](../../06_llm_training_and_lora/05_lora_peft_finetune/WALKTHROUGH.md)). It trained a model to predict the next token of some text. This module asks the next question: how does a model that only predicts text become an *assistant*, and how do you prove it got better?

## The pain point
A pretrained **base model** (gpt2, Qwen2.5-0.5B, Llama base) does not answer questions. Ask it "What is the capital of France?" and it continues the document: more questions, more sentences. Worse, a chat model is trained on a very specific *string format* with special tokens (`<|im_start|>`, `<|im_end|>`). If you feed it a different format at inference, quality silently drops; if you train with a different format than you serve, the model learns the wrong thing. And nobody tells you which of the tokens in that string should be penalised by the loss.

## The spark
Treat "making an assistant" as a pipeline of small, well-defined stages, each with its own data format and loss: pretraining, mid-training, SFT, preference tuning, RL, evaluation. And treat the **chat template** as the contract between data, training and serving: one function (`apply_chat_template`) turns a list of messages into the exact string, and the same string layout also tells you which tokens carry the loss.

## Goal
Print the six-stage map, then *see* what a real chat template (Qwen2.5's) produces: the exact string, the token ids, the generation prompt, the completion-only loss mask, and the four dataset formats (prompt/completion, messages, preference pair, verifiable). Finally show that a base model (gpt2) continues text instead of answering. Only the **tokenizer** of Qwen2.5-0.5B is cached on this machine; its weights are not, so nothing here runs Qwen itself.

## The theory (read this before the interview)
**The pipeline (Nov 2026 practice):**
```
pretraining  -> mid-training -> SFT -> preference tuning -> RL (RLHF / RLVR) -> evaluation
 next-token     same loss,      CE on     DPO-style loss on    policy gradient vs a     benchmarks, judges,
 CE on web      curated /       (prompt   (chosen, rejected)    reward model or a        human arenas
 text           synthetic /     -> good   pairs                verifier (code)
                long context    answer)
 base model     stronger base   instruct  aligned model        reasoning / agent model  a scorecard
```
- **Pretraining** gives knowledge and language. **Mid-training** (a stage many labs now run between pretraining and SFT) uses the same loss on a more curated mix. **SFT** (supervised fine-tuning, instruction tuning) teaches format and instruction following. **Preference tuning** teaches taste (tone, helpfulness, safety). **RL** with a verifier teaches checkable skills (maths, code); reasoning models are made this way. Evaluation closes the loop.
- This is current industry practice at a high level. Exact recipes differ per lab and are often not published, so say "typically" in an interview.

**Base vs instruct vs reasoning model:** base = pretrained only (autocomplete). Instruct/chat = SFT and usually preference tuning on chat data. Reasoning = RL-trained to emit a long chain of thought before the answer.

**Chat template** = a Jinja template stored with the tokenizer. For Qwen2.5 (ChatML-style) a conversation becomes
```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
<|im_start|>user\nWhat is the capital of France?<|im_end|>\n
<|im_start|>assistant\nParis.<|im_end|>\n
```
`<|im_start|>` and `<|im_end|>` are single **special tokens** (one id each), not text the tokenizer splits. `<|im_end|>` is the end-of-turn marker: the model learns to emit it to stop. `add_generation_prompt=True` appends the empty assistant header `<|im_start|>assistant\n` so the model knows where to start writing.

**Completion-only loss mask:** cross-entropy is computed only on the assistant tokens. Everything before them (system, user, headers) gets label `-100`, which `F.cross_entropy(..., ignore_index=-100)` skips. The next step ([step 02](../02_sft_from_scratch_with_loss_masking/WALKTHROUGH.md)) trains with exactly this mask.

**Four dataset formats**, one per stage:
| Format | Fields | Used by |
|---|---|---|
| prompt/completion | `prompt`, `completion` | SFT (simple) |
| messages (chat) | `messages=[{role, content}, ...]` | SFT on chat models |
| preference pair | `prompt`, `chosen`, `rejected` | reward model, DPO |
| verifiable (RLVR) | `prompt`, `ground_truth` | RL with a verifier |

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| SFT | supervised fine-tuning, instruction tuning, behaviour cloning | near | Cross-entropy on good answers; "instruction tuning" names the data, SFT names the method |
| Base model | pretrained model, foundation model | near | Never saw chat data; just continues text |
| Instruct / chat model | assistant model, aligned model | near | SFT (and usually preference tuning) applied on top of a base |
| Chat template | prompt template, ChatML (one specific style) | no | Template is the mechanism; ChatML, Llama-3 and Gemma styles are different templates |
| `<|im_end|>` | end-of-turn token, EOS (for chat) | near | The turn terminator; the tokenizer's `eos_token` here is `<|endoftext|>`, a different token |
| Generation prompt | assistant header, `add_generation_prompt` | yes | The empty assistant turn opener appended at inference |
| Loss mask | label masking, completion-only loss, `-100` labels | yes | Positions ignored by cross-entropy |
| Mid-training | continued pretraining, annealing stage | near | Same loss, curated data, before SFT |
| RLHF / RLVR | RL from human feedback / verifiable rewards | no | Differ only in where the reward comes from (step 06) |

## Run it
```bash
cd 17_finetuning_alignment_and_benchmarks/01_post_training_map_and_chat_templates
python main.py
# from repo root:
HF_HUB_OFFLINE=1 .venv/bin/python 17_finetuning_alignment_and_benchmarks/01_post_training_map_and_chat_templates/main.py
# knob: BASE_LM (default gpt2)
```
About 15 s of compute on CPU (plus model loading); wall time varies by machine and is longer if another heavy job holds the lock.

## Code, step by step

Lines 1–15 of `main.py` are the header docstring (GOAL / RUN / EXPECTED OUTPUT / NEEDS) already summarised above; the walk starts after it.

### 0. Imports and settings  (`main.py` lines 16–22)
```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
TOK_NAME = "Qwen/Qwen2.5-0.5B"      # only the TOKENIZER of Qwen is cached offline (the weights are not)
BASE_LM = os.environ.get("BASE_LM", "gpt2")   # a real base (not instruct) model that we can run on a CPU
```
- **In simple words:** Load the tools and name the two models: Qwen's tokenizer (for the template) and gpt2 (a real base model we can run).
- **Exactly:** `AutoTokenizer` and `AutoModelForCausalLM` are Hugging Face factory classes that read the model's `config.json` and return the right concrete class. `BASE_LM` can be overridden through the environment, e.g. `BASE_LM=distilgpt2`.
- **Why it is here:** The template lives in the Qwen tokenizer files, but running Qwen needs weights we do not have offline, so a second, smaller model demonstrates base-model behaviour.

### 1. The post-training map as data  (`main.py` lines 24–35)
```python
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
```
- **In simple words:** The whole module as a six-row table: what signal drives each stage and what comes out.
- **Exactly:** `STAGES` is a list of 4-tuples of strings. The f-string `{s[0]:<20}` left-aligns a string in a 20-character column. No computation happens; it is a map.
- **Why it is here:** Every later step is one row of this table: step 02 = row 3, step 04/05 = row 4, step 06 = row 5, steps 08-11 = row 6.

### 2. Load the tokenizer and build a chat prompt  (`main.py` lines 37–43)
```python
# ## 2. Load the tokenizer and build a chat prompt
tok = AutoTokenizer.from_pretrained(TOK_NAME)
messages = [{"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris."}]
text = tok.apply_chat_template(messages, tokenize=False)        # a plain string, no ids yet
print("\n--- chat template output (repr) ---")
print(repr(text))
```
- **In simple words:** Hand the tokenizer a two-turn conversation and ask it for the exact text the model would be trained on.
- **Exactly:** `AutoTokenizer.from_pretrained` reads `tokenizer.json`, `tokenizer_config.json` (which contains the Jinja `chat_template`) and the special-token map from the local cache. `apply_chat_template(messages, tokenize=False)` renders the Jinja template over the list of `{"role", "content"}` dicts and returns a `str`. We passed no system message, yet the output starts with `<|im_start|>system\nYou are a helpful assistant.<|im_end|>`: Qwen's template inserts a default system prompt when none is given. `repr` shows the invisible `\n` characters.
- **Why it is here:** This is the contract. Whatever string your SFT data used is the string the model expects at serving time.

### 3. Generation prompt: where the model should start writing  (`main.py` lines 45–48)
```python
# ## 3. Generation prompt: where the model should start writing
prompt = tok.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
print("\n--- same, user turn only + add_generation_prompt=True ---")
print(repr(prompt))
```
- **In simple words:** At inference the conversation stops after the user's message, and we add the opening of the assistant's turn so the model continues from there.
- **Exactly:** `messages[:1]` is just the user turn. `add_generation_prompt=True` makes the template append `<|im_start|>assistant\n` and nothing else. The result ends with a newline and no `<|im_end|>`.
- **Why it is here:** Used again in section 5 to find where the assistant's tokens start, and in section 7 to feed gpt2 the same string.

### 4. Exact token ids and pieces  (`main.py` lines 50–54)
```python
# ## 4. Exact token ids and pieces
ids = tok(text, add_special_tokens=False).input_ids              # list[int]; the template already holds the specials
print(f"\nn_tokens={len(ids)}  eos_token={tok.eos_token!r}  pad_token={tok.pad_token!r}  vocab={len(tok)}")
print("id -> piece:")
print("  " + "  ".join(f"{i}:{tok.decode([i])!r}" for i in ids[:14]) + " ...")
```
- **In simple words:** Turn the string into numbers and show the first 14 so you can see that `<|im_start|>` is one token.
- **Exactly:** `tok(text, ...)` returns a `BatchEncoding`; `.input_ids` is a Python `list[int]`. `add_special_tokens=False` stops the tokenizer from adding its own BOS/EOS on top of the specials already written in the string (Qwen adds none by default, but the habit prevents double specials in other models). `tok.decode([i])` converts one id back to text. `len(tok)` counts the base vocabulary plus added special tokens.
- **Why it is here:** The model never sees text, only ids. `<|im_start|>` = 151644 and `<|im_end|>` = 151645 are single ids at the end of the vocabulary: they are *added* tokens, so a tokenizer that did not know them would split them into many pieces.

### 5. Which tokens carry the SFT loss (completion-only mask)  (`main.py` lines 56–62)
```python
# ## 5. Which tokens carry the SFT loss (completion-only mask)
prompt_ids = tok(prompt, add_special_tokens=False).input_ids     # everything the user side produced
full_ids = ids
# The assistant turn = tokens after the generation prompt. Mask prompt positions with -100 (ignored by cross entropy).
labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
print(f"\nprompt tokens={len(prompt_ids)}  trained tokens={sum(l != -100 for l in labels)}")
print("TRAIN tokens:", [tok.decode([l]) for l in labels if l != -100])
```
- **In simple words:** Everything up to and including `<|im_start|>assistant\n` is "the question"; only what comes after is "the answer" that the model should learn to produce.
- **Exactly:** `prompt_ids` is the tokenised generation prompt (26 tokens). Because the full conversation string *starts with* the generation prompt string, the answer tokens are `full_ids[26:]`. `labels` has the same length as `ids`: `-100` for the first 26 positions, the real ids afterwards. `-100` is the value `torch.nn.functional.cross_entropy` ignores by default (`ignore_index=-100`). The trained tokens are `Paris`, `.`, `<|im_end|>` and the final `\n`.
- **Why it is here:** This is the exact labelling rule step 02 implements for a plain prompt/completion pair. Training on the user's tokens too wastes capacity and teaches the model to *ask* questions. Note that `<|im_end|>` is trained: that is how the model learns to stop.

### 6. The four dataset formats  (`main.py` lines 64–73)
```python
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
```
- **In simple words:** The same arithmetic fact written four ways, one for each kind of training.
- **Exactly:** Plain Python dicts. `FORMATS.items()` yields `(name, row)`. Note the leading space in `" 4"`: GPT-style tokenizers attach the space to the next word, so a completion that follows `A:` is tokenised as `" 4"`, not `"4"`.
- **Why it is here:** Know which fields each trainer needs: SFT wants `prompt/completion` or `messages`; reward-model and DPO training want `chosen/rejected`; GRPO wants a prompt plus something a verifier can check.

### 7. A base model does not "answer": it continues text  (`main.py` lines 75–85)
```python
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
```
- **In simple words:** Ask gpt2 the question two ways and see that it just keeps writing; it has no idea it should answer.
- **Exactly:** `AutoModelForCausalLM.from_pretrained("gpt2")` loads the weights (an LM head over a 50,257-token vocabulary); `.eval()` switches off dropout. `generate(..., do_sample=False)` is **greedy decoding** (argmax at every step). It loops: forward pass, take the argmax of the last position's logits, append, repeat `max_new_tokens=24` times, using the KV cache internally. `out[0, L:]` slices off the prompt so only new tokens are decoded. `pad_token_id` is set only to silence a warning (gpt2 has no pad token). The second call feeds gpt2 the *Qwen-formatted* string: gpt2 never saw `<|im_start|>`, so it treats it as odd text.
- **Why it is here:** It shows the pain point with real output. In the run below, gpt2 answers the raw question with a repeated document pattern and echoes the Qwen header string for the chat-formatted one. (Interesting detail: for the raw question gpt2's continuation does contain "Paris"; it is not that the model lacks the fact, it lacks the *behaviour* of answering once and stopping.)

### 8. TRY IT / BREAK IT comments  (`main.py` lines 87–89)
```python
# TRY IT: pass a system message ({"role": "system", "content": "Answer in French."}) and print the template again.
# BREAK IT: tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) on the FULL conversation ends with an
#           empty assistant header, so a model trained on it would learn to answer after an answer. Print it and look.
```
- **In simple words:** Two suggested experiments; both are run in the Experiments and Debugging sections below.
- **Exactly:** Comments only, no code.
- **Why it is here:** The module convention is that every file ends with a TRY IT and a BREAK IT.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `tok.apply_chat_template(msgs, tokenize=False)` | render messages to a string with the model's template |
| `add_generation_prompt=True` | append the empty assistant header so generation can start |
| `tok(text, add_special_tokens=False).input_ids` | token ids without extra BOS/EOS |
| `[-100] * n + ids` | label list: `n` ignored positions, then real targets |
| `f"{x:<20}"` | left-align in a 20-character field |
| `out[0, L:]` | batch row 0, positions from `L` onward (only the new tokens) |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `AutoTokenizer.from_pretrained` | loads a tokenizer from the local HF cache | reads `tokenizer.json`, `tokenizer_config.json`, special-token map; `HF_HUB_OFFLINE=1` forbids any download |
| `apply_chat_template` | messages -> prompt string (or ids) | runs the Jinja template stored in `tokenizer_config.json` |
| `AutoModelForCausalLM.from_pretrained` | loads transformer weights with a next-token head | builds the architecture from `config.json`, then loads `safetensors` weights |
| `model.generate` | autoregressive decoding | repeated forward pass + argmax (greedy) with a KV cache of past keys and values |
| `tok.decode` | ids -> text | maps ids to byte-level pieces and joins them |

## Execution trace (real run)
```
stage               signal / loss                                       output                  teaches
1 pretraining       next-token loss on trillions of web/code tokens     base model              knowledge, language
2 mid-training      same loss, curated/synthetic/long-context data      stronger base           reasoning, long context
3 SFT               cross-entropy on (prompt -> good answer)            instruct (chat) model   format, following instructions
4 preference tuning DPO-style loss on (chosen, rejected) pairs          aligned model           tone, helpfulness, safety
5 RL (RLHF/RLVR)    policy gradient vs reward model or verifier         reasoning/agent model   verifiable skills (math, code)
6 evaluation        benchmarks + judges + human arenas                  a scorecard             decide ship / iterate

--- chat template output (repr) ---
'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\nParis.<|im_end|>\n'

--- same, user turn only + add_generation_prompt=True ---
'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n'

n_tokens=30  eos_token='<|endoftext|>'  pad_token='<|endoftext|>'  vocab=151665
id -> piece:
  151644:'<|im_start|>'  8948:'system'  198:'\n'  2610:'You'  525:' are'  264:' a'  10950:' helpful'  17847:' assistant'  13:'.'  151645:'<|im_end|>'  198:'\n'  151644:'<|im_start|>'  872:'user'  198:'\n' ...

prompt tokens=26  trained tokens=4
TRAIN tokens: ['Paris', '.', '<|im_end|>', '\n']

--- dataset formats ---
prompt/completion   {'prompt': 'Q: 2+2?\nA:', 'completion': ' 4'}
messages (chat)     {'messages': [{'role': 'user', 'content': 'What is the capital of France?'}, {'role': 'assistant', 'content': 'Paris.'}]}
preference pair     {'prompt': 'Q: 2+2?\nA:', 'chosen': ' 4', 'rejected': ' 5'}
verifiable (RLVR)   {'prompt': 'Q: 2+2?\nA:', 'ground_truth': '4'}

--- gpt2 (base model) greedy continuation ---
raw question  : '\n\nThe capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe capital of France'
chat-formatted: 'What is the capital of France?<|im_end|>\n<|im_start|>assistant'
```
(A one-line torch "Redirects are currently not supported in MacOs" note and a "Loading weights" progress bar also appear on stderr; omitted.)
- The default system message is injected by the template (`You are a helpful assistant.`) even though our list had none: 30 tokens in total, of which the first 26 are the prompt side.
- `<|im_start|>` is token 151644 and `<|im_end|>` 151645: single ids, so the whole turn structure costs only a handful of tokens. The `eos_token` of this tokenizer is `<|endoftext|>`, *not* `<|im_end|>`: do not assume they are the same.
- Only 4 of 30 positions carry the loss (`Paris`, `.`, `<|im_end|>`, `\n`). In real SFT data with long user prompts the masked share is far larger.
- gpt2 repeats "The capital of France is Paris." on the raw question: the fact is there but the behaviour (answer once, stop) is not. Given the Qwen-formatted prompt it just re-emits the header text, because it has never seen those tokens as structure.
- The raw `Loading weights` bar and Redirect note are logging noise, not part of the program's output.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| (measured) add `{"role": "system", "content": "Answer in French."}` in front of the messages | the string starts `<|im_start|>system\nAnswer in French.<|im_end|>\n<|im_start|>user...`; the default "You are a helpful assistant." is gone | Qwen's template only injects its default system turn when you give none |
| (measured) `tok.apply_chat_template(messages[:1], tokenize=False)` without `add_generation_prompt` | ends after `<|im_end|>\n` of the user turn, with no `<|im_start|>assistant` header | the generation prompt is opt-in; without it the model would not know it is the assistant's turn |
| (measured) `tok("<|im_start|>").input_ids` | `[151644]`: one token, length 1 | the special token is matched as a whole before byte-pair splitting |
| (not run) `BASE_LM=distilgpt2 python main.py` | another base model, also continuing text rather than answering (exact words will differ) | answering is a trained behaviour, not a property of model size |

## Debugging practice
**BREAK IT:** call `apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` on the *full* conversation (the one that already contains the assistant answer). Real output of that call:
```
'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\nParis.<|im_end|>\n<|im_start|>assistant\n'
```
How to read it: no exception is raised, which is what makes it dangerous. The string ends with a second, empty `<|im_start|>assistant\n`. If you used it as SFT data, the loss mask would be computed from a wrong boundary and the model would see an assistant turn that follows an assistant turn. The symptom in practice is a fine-tuned model that sometimes answers twice or never emits `<|im_end|>` cleanly. The fix is a rule: `add_generation_prompt=True` only at inference on a conversation that ends with a user turn; for training render the complete conversation with it off. Always `print(repr(text))` once per dataset.

## Common mistakes
- Hand-writing the chat string instead of calling `apply_chat_template`: one missing newline or token and train/serve formats diverge.
- Training with a different template than the one used at inference (or the model's built-in one).
- Forgetting to train the end-of-turn token, so the fine-tuned model never stops.
- Applying `add_generation_prompt=True` to a conversation that already contains the assistant answer (the BREAK IT).
- Assuming `eos_token` is the end-of-turn token; in ChatML-style tokenizers it is not.
- Calling a base model "broken" because it does not answer; it was never trained to.

## Rebuild checklist
1. A list of messages and `tok.apply_chat_template(messages, tokenize=False)`.
2. The generation-prompt variant on the user turn only (`add_generation_prompt=True`).
3. Token ids with `add_special_tokens=False` and a printed id-to-piece list.
4. Labels: `-100` for the prompt tokens (length of the generation prompt), real ids for the answer.
5. The four dataset-format dicts.
6. A greedy `generate` of a base model, slicing off the prompt tokens.

## Say it aloud
"A base model only continues text; post-training turns it into an assistant through SFT, preference tuning and RL, and evaluation closes the loop. The chat template is the contract between data, training and serving: it wraps turns in special tokens like `<|im_start|>` and `<|im_end|>`, and `add_generation_prompt` opens the assistant turn at inference. For SFT I compute the loss only on assistant tokens by labelling everything else `-100`, including training the end-of-turn token so the model learns to stop. Each stage has its own data format: prompt/completion or messages for SFT, chosen/rejected for preference tuning, and prompt plus ground truth for verifiable RL."

## The new problem (cliffhanger)
You now know *which* tokens should carry the loss. Next: write the training loop yourself, apply the `-100` mask to real batches (with padding), fine-tune distilgpt2, and measure before and after on reviews that include words never seen in training. On to [step 02](../02_sft_from_scratch_with_loss_masking/WALKTHROUGH.md).
