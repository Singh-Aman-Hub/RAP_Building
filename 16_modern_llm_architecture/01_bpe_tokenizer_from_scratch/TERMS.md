# Terms — Byte-level BPE tokenizer

Folder: `16_modern_llm_architecture/01_bpe_tokenizer_from_scratch` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [16 README](../README.md)

**Terms in this folder:** [Byte-pair encoding (BPE)](#byte-pair-encoding-bpe) · [Byte-level BPE](#byte-level-bpe) · [Merge rule](#merge-rule) · [Vocabulary](#vocabulary) · [Compression ratio (bytes per token)](#compression-ratio-bytes-per-token) · [encode and decode](#encode-and-decode) · [Counter and most_common](#counter-and-most_common) · [UTF-8 and bytes](#utf-8-and-bytes) · [Special tokens](#special-tokens) · [Chat template](#chat-template) · [AutoTokenizer.from_pretrained](#autotokenizerfrom_pretrained) · [SentencePiece and Unigram](#sentencepiece-and-unigram) · [tiktoken](#tiktoken) · [Embedding table size](#embedding-table-size)

---

## Byte-pair encoding (BPE)  `(concept)`
**1. Definition:** A subword algorithm that starts from single symbols and repeatedly merges the most frequent adjacent pair into a new symbol until a target vocabulary size is reached.
**2. In plain words:** Count neighbouring pairs, glue the most common one, repeat. After 300 merges on our text, 20,001 bytes became 8,960 tokens (2.23 bytes per token).
**3. Example to quote:** > "BPE learns its vocabulary from data: frequent character sequences become single tokens."
**Seen in:** `main.py`, training loop

## Byte-level BPE  `(concept)`
**1. Definition:** BPE whose base alphabet is the 256 byte values of UTF-8, so every possible string is representable with no unknown token.
**2. In plain words:** Because it works on bytes, emoji, Hindi and code all work; rare characters just cost several byte tokens.
**3. Example to quote:** > "Byte-level BPE can never produce an unknown token."
**Seen in:** `main.py`, `data = list(text.encode('utf-8'))`

## Merge rule  `(concept)`
**1. Definition:** A learned pair (a, b) -> new id; the ordered list of merge rules is what a BPE tokenizer is.
**2. In plain words:** The first rule says 'two spaces become token 256'. Rules are applied in the order they were learned.
**3. Example to quote:** > "A BPE tokenizer is its merge list plus the base bytes."
**Seen in:** `main.py`, `merges` dict

## Vocabulary  `(concept)`
**1. Definition:** The set of tokens a model knows, each with an integer id; its size sets the rows of the embedding table.
**2. In plain words:** Our toy: 256 + 300 = 556 tokens; gpt2: 50,257; Qwen2.5: 151,665 real tokens.
**3. Example to quote:** > "Vocabulary size trades shorter sequences against a bigger embedding table."
**Seen in:** `main.py`, sections 3 and 5

## Compression ratio (bytes per token)  `(concept)`
**1. Definition:** Average number of input bytes represented by one token; higher means fewer tokens for the same text.
**2. In plain words:** 2.23 for our 300 merges, 3.61 for 1,000 merges, about 4 for good English tokenizers.
**3. Example to quote:** > "More merges mean fewer tokens, with diminishing returns."
**Seen in:** `main.py`, training print

## encode and decode  `(concept)`
**1. Definition:** encode maps text to token ids by applying merges in learned order; decode maps ids back to bytes and then to a string.
**2. In plain words:** Encode: text to numbers. Decode: numbers to text. They must be inverses.
**3. Example to quote:** > "A correct tokenizer round-trips: decode(encode(s)) == s."
**Seen in:** `main.py`, `encode`, `decode`

## Counter and most_common  `(library)`
**1. Definition:** collections.Counter counts hashable items; most_common(n) returns the n largest counts.
**2. In plain words:** It tells us which pair of tokens is most frequent.
**3. Example to quote:** > "Counter(zip(ids, ids[1:])) counts all adjacent pairs."
**Seen in:** `main.py`, `pair_counts`

## UTF-8 and bytes  `(concept)`
**1. Definition:** UTF-8 encodes each Unicode character as 1 to 4 bytes; str.encode('utf-8') returns the bytes.
**2. In plain words:** 'e' is 1 byte, an accented e is 2, an emoji is 4.
**3. Example to quote:** > "The 'naive cafe' test string is 17 bytes but fewer characters."
**Seen in:** `main.py`, section 1

## Special tokens  `(concept)`
**1. Definition:** Reserved token ids that are not produced by merges, such as <|endoftext|>, <|im_start|> and <|im_end|>.
**2. In plain words:** Markers that tell the model where a document or a chat turn starts and ends.
**3. Example to quote:** > "Special tokens are single ids: '<|im_start|>' is id 151644 in Qwen."
**Seen in:** `main.py`, section 6

## Chat template  `(concept)`
**1. Definition:** A Jinja template stored with the tokenizer that converts a list of role/content messages into the exact prompt string the model was trained on.
**2. In plain words:** The same messages need different markers for Qwen (ChatML) and Llama; using the wrong template quietly degrades answers.
**3. Example to quote:** > "apply_chat_template(add_generation_prompt=True) ends the prompt with the assistant header."
**Seen in:** `main.py`, `qwen.apply_chat_template`

## AutoTokenizer.from_pretrained  `(library)`
**1. Definition:** Hugging Face call that loads the tokenizer files (tokenizer.json or vocab.json + merges.txt) for a model name or path.
**2. In plain words:** One line to get the real tokenizer of any model, offline if cached.
**3. Example to quote:** > "HF_HUB_OFFLINE=1 makes it use only the local cache."
**Seen in:** `main.py`, section 5

## SentencePiece and Unigram  `(concept)`
**1. Definition:** SentencePiece is a tokenizer library working on raw text with the space as a symbol; its Unigram algorithm prunes a large candidate vocabulary by likelihood instead of merging.
**2. In plain words:** Another way to build subwords; used by T5 and many multilingual models.
**3. Example to quote:** > "SentencePiece can use BPE or Unigram inside."
**Seen in:** `WALKTHROUGH.md`, theory

## tiktoken  `(library)`
**1. Definition:** OpenAI's fast BPE tokenizer library; it encodes text with OpenAI's published vocabularies. Not installed in this repo.
**2. In plain words:** A fast encoder, not a new algorithm.
**3. Example to quote:** > "tiktoken is to GPT models what AutoTokenizer is to Hugging Face models."
**Seen in:** `WALKTHROUGH.md`, theory

## Embedding table size  `(concept)`
**1. Definition:** vocab_size x d_model parameters; the input embedding (and, if untied, the output head).
**2. In plain words:** Qwen2.5-0.5B: 151,936 x 896 = 136.1M, over a quarter of the model.
**3. Example to quote:** > "A big vocabulary is cheap in a 70B model and expensive in a 0.5B one."
**Seen in:** `main.py`, section 7

