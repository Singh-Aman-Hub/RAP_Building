# Terms — The post-training map and chat templates

Folder: `17_finetuning_alignment_and_benchmarks/01_post_training_map_and_chat_templates` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [17 README](../README.md)

**Terms in this folder:** [Post-training](#post-training) · [Pretraining and mid-training](#pretraining-and-mid-training) · [Base vs instruct vs reasoning model](#base-vs-instruct-vs-reasoning-model) · [Chat template](#chat-template) · [Special tokens (<|im_start|>, <|im_end|>)](#special-tokens-im_start-im_end) · [add_generation_prompt](#add_generation_prompt) · [System prompt](#system-prompt) · [Prompt/completion, messages, preference and verifiable formats](#promptcompletion-messages-preference-and-verifiable-formats) · [Loss mask (labels = -100)](#loss-mask-labels---100) · [EOS token and pad token](#eos-token-and-pad-token) · [Tokenizer vs model weights](#tokenizer-vs-model-weights) · [Greedy decoding](#greedy-decoding) · [Evaluation as the last stage](#evaluation-as-the-last-stage) · [apply_chat_template](#apply_chat_template)

---

## Post-training  `(concept)`
**1. Definition:** Every stage after pretraining: mid-training, SFT, preference tuning, reinforcement learning, and evaluation.
**2. In plain words:** Everything that turns a text-continuing model into an assistant.
**3. Example to quote:** > "Pretraining gives knowledge; post-training gives behaviour."
**Seen in:** `main.py`, STAGES table

## Pretraining and mid-training  `(concept)`
**1. Definition:** Pretraining: next-token cross-entropy on a huge corpus. Mid-training: the same loss on curated, synthetic or long-context data before SFT.
**2. In plain words:** Reading everything, then reading the good parts.
**3. Example to quote:** > "Pretraining reads the whole web with a next-token loss; mid-training runs the same loss on a curated mix before SFT starts."
**Seen in:** `main.py`, STAGES table

## Base vs instruct vs reasoning model  `(concept)`
**1. Definition:** Base: pretrained only, continues text. Instruct: SFT (and usually preference tuned) on chat data. Reasoning: further RL-trained to produce long chains of thought before answering.
**2. In plain words:** Autocomplete, assistant, and thinker.
**3. Example to quote:** > "gpt2 (a base model) answered the raw question by continuing the text, not by answering."
**Seen in:** `main.py`, section 7

## Chat template  `(concept)`
**1. Definition:** A Jinja template in the tokenizer that turns messages into the exact string, special tokens included, the model was trained on.
**2. In plain words:** The envelope format.
**3. Example to quote:** > "Qwen2.5 wraps each turn as <|im_start|>role\n...<|im_end|>\n and adds a default system prompt."
```python
tok.apply_chat_template(messages, tokenize=False)
```
**Seen in:** `main.py`, section 2

## Special tokens (<|im_start|>, <|im_end|>)  `(concept)`
**1. Definition:** Reserved vocabulary entries that mark turn boundaries; each is one token id.
**2. In plain words:** Punctuation for conversation structure.
**3. Example to quote:** > "<|im_start|> is id 151644 and <|im_end|> is id 151645 in the Qwen2.5 tokenizer."
```python
tok.decode([151644])
```
**Seen in:** `main.py`, section 4

## add_generation_prompt  `(syntax)`
**1. Definition:** apply_chat_template flag that appends the open assistant header so generation starts in the right place.
**2. In plain words:** The 'your turn' marker.
**3. Example to quote:** > "Use it at inference; do not use it for already-finished training conversations."
```python
tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
```
**Seen in:** `main.py`, section 3

## System prompt  `(concept)`
**1. Definition:** A first message with role system that sets behaviour; Qwen2.5's template inserts 'You are a helpful assistant.' if none is given.
**2. In plain words:** Standing instructions.
**3. Example to quote:** > "The Qwen template injects 'You are a helpful assistant.' as a system turn even when my message list has none."
**Seen in:** `main.py`, section 2

## Prompt/completion, messages, preference and verifiable formats  `(concept)`
**1. Definition:** The four dataset layouts: {prompt, completion}; {messages}; {prompt, chosen, rejected}; {prompt, ground_truth}.
**2. In plain words:** Which stage uses which shape of data.
**3. Example to quote:** > "SFT uses the first two, DPO the third, RLVR the fourth."
```python
{'prompt': 'Q', 'chosen': 'A', 'rejected': 'B'}
```
**Seen in:** `main.py`, section 6

## Loss mask (labels = -100)  `(concept)`
**1. Definition:** Positions whose label is -100 are ignored by cross_entropy(ignore_index=-100).
**2. In plain words:** Do not grade this part.
**3. Example to quote:** > "For the 30-token conversation, 26 prompt tokens are masked and 4 are trained: 'Paris', '.', <|im_end|>, '\n'."
```python
labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
```
**Seen in:** `main.py`, section 5

## EOS token and pad token  `(concept)`
**1. Definition:** EOS marks end of sequence; pad fills batches. Qwen2.5 uses <|endoftext|> for both, while chat turns end with <|im_end|>.
**2. In plain words:** Stop signal and filler.
**3. Example to quote:** > "Align the EOS token with the chat template (TRL SFTConfig(eos_token='<|im_end|>') for Qwen2.5 base)."
```python
tok.eos_token, tok.pad_token
```
**Seen in:** `main.py`, section 4

## Tokenizer vs model weights  `(concept)`
**1. Definition:** The tokenizer is a separate small artefact from the weights; only the Qwen2.5 tokenizer is cached offline here.
**2. In plain words:** Dictionary vs brain.
**3. Example to quote:** > "Template inspection needs only the tokenizer."
```python
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
```
**Seen in:** `main.py`, section 2

## Greedy decoding  `(concept)`
**1. Definition:** Always pick the most likely next token (do_sample=False).
**2. In plain words:** No randomness.
**3. Example to quote:** > "Used so the base-model demonstration is reproducible."
```python
model.generate(**enc, do_sample=False)
```
**Seen in:** `main.py`, section 7

## Evaluation as the last stage  `(concept)`
**1. Definition:** Benchmarks, judges and human arenas decide ship or iterate.
**2. In plain words:** The report card.
**3. Example to quote:** > "Steps 08-11 cover it."
**Seen in:** `main.py`, STAGES table

## apply_chat_template  `(library)`
**1. Definition:** Tokenizer method that renders messages with the model's template (tokenize=False returns the string).
**2. In plain words:** Format the conversation.
**3. Example to quote:** > "I never hand-write the chat string: apply_chat_template renders the exact format the model was trained on."
**Seen in:** `main.py`, section 2
