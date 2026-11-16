# Terms — Contextual Retrieval

Folder: `19_chunking_deep_dive/06_contextual_retrieval` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [19 README](../README.md)

**Terms in this folder:** [Contextual retrieval](#contextual-retrieval) · [Contextual embeddings vs contextual BM25](#contextual-embeddings-vs-contextual-bm25) · [Situating prompt](#situating-prompt) · [Prompt caching](#prompt-caching) · [MockLLM with rules](#mockllm-with-rules) · [Heuristic context](#heuristic-context) · [Offset-preserving blanking](#offset-preserving-blanking) · [AttributeError: 'NoneType' has no attribute 'group'](#attributeerror-nonetype-has-no-attribute-group)

---

## Contextual retrieval  `(concept)`
**1. Definition:** An LLM writes 1-2 sentences situating each chunk in its document; the text is prepended before embedding and BM25 indexing (Anthropic, Sept 2024).
**2. In plain words:** A note on every chunk saying what it is about and where it is from.
**3. Example to quote:** > "Contextual embeddings plus contextual BM25 cut failed top-20 retrievals by 49% in Anthropic's tests."
**Seen in:** main.py, situate

## Contextual embeddings vs contextual BM25  `(concept)`
**1. Definition:** Embeddings: context + chunk is embedded. BM25: context + chunk is tokenised for the sparse index.
**2. In plain words:** Dense learns the meaning, sparse matches the exact words in the context.
**3. Example to quote:** > "Contextual embeddings vs contextual BM25: Dense learns the meaning, sparse matches the exact words in the context."
**Seen in:** main.py, section B

## Situating prompt  `(concept)`
**1. Definition:** A prompt with <document> and <chunk> tags asking for a short context only.
**2. In plain words:** Give the model the whole document and the chunk.
**3. Example to quote:** > "Situating prompt: Give the model the whole document and the chunk."
**Seen in:** main.py, PROMPT

## Prompt caching  `(concept)`
**1. Definition:** The provider stores the repeated prefix (the document); later calls read it cheaper. Multipliers used here: write 1.25x, read 0.10x of the input price (check current pricing).
**2. In plain words:** Pay once to send the document, then little.
**3. Example to quote:** > "3.1x fewer input-token equivalents in our run."
**Seen in:** main.py, section C

## MockLLM with rules  `(library)`
**1. Definition:** shared/llm_client.MockLLM(rules=[(regex, callable)]) answers deterministically; LLM_BACKEND=anthropic switches to a real model.
**2. In plain words:** A stand-in so the module runs offline.
**3. Example to quote:** > "MockLLM with rules: A stand-in so the module runs offline."
```python
MockLLM(rules=[(r'succinct context', _mock_rule)])
```
**Seen in:** main.py, MOCK

## Heuristic context  `(concept)`
**1. Definition:** A rule-based stand-in: document title + section path + the first words of the section (or the previous sentence).
**2. In plain words:** What a real LLM writes, in miniature.
**3. Example to quote:** > "Mock results are not LLM results."
**Seen in:** main.py, heuristic_context

## Offset-preserving blanking  `(concept)`
**1. Definition:** Replacing heading lines by spaces keeps every character offset valid while removing the information.
**2. In plain words:** Simulates text extracted without headings.
**3. Example to quote:** > "Offset-preserving blanking: Simulates text extracted without headings."
**Seen in:** main.py, section B2

## AttributeError: 'NoneType' has no attribute 'group'  `(error)`
**1. Definition:** re.search returned None because the <chunk> tags are missing from the prompt.
**2. In plain words:** The regex found nothing.
**3. Example to quote:** > "AttributeError: 'NoneType' has no attribute 'group': The regex found nothing."
**Seen in:** main.py, BREAK IT
