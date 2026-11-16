# Terms — Propositions and Agentic Chunking

Folder: `19_chunking_deep_dive/08_propositions_and_agentic_chunking` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [19 README](../README.md)

**Terms in this folder:** [Proposition](#proposition) · [Decontextualisation](#decontextualisation) · [Agentic chunking](#agentic-chunking) · [Ingestion cost (LLM calls and tokens)](#ingestion-cost-llm-calls-and-tokens) · [Mock LLM rules](#mock-llm-rules) · [extract_json](#extract_json) · [Source span of a proposition](#source-span-of-a-proposition) · [Lexical overlap as topic proxy](#lexical-overlap-as-topic-proxy)

---

## Proposition  `(concept)`
**1. Definition:** An atomic, self-contained factual statement; pronouns replaced by their referent (Dense X Retrieval, arXiv 2312.06648).
**2. In plain words:** One fact per chunk.
**3. Example to quote:** > "Each proposition must make sense alone."
**Seen in:** main.py, rule_propositions

## Decontextualisation  `(concept)`
**1. Definition:** Rewriting a sentence so that it needs no surrounding text: 'It can start...' becomes 'The primary caregiver can start...'.
**2. In plain words:** Replace 'it' by the noun.
**3. Example to quote:** > "Decontextualisation: Replace 'it' by the noun."
**Seen in:** main.py, PRONOUN

## Agentic chunking  `(concept)`
**1. Definition:** An LLM decides boundaries or grouping, here SAME/NEW per sentence.
**2. In plain words:** A model as editor.
**3. Example to quote:** > "Agentic chunking: A model as editor."
**Seen in:** main.py, agentic_chunks

## Ingestion cost (LLM calls and tokens)  `(concept)`
**1. Definition:** Propositions: 140 calls, 5.3x the corpus tokens in+out. Agentic: one call per sentence (362).
**2. In plain words:** What you pay before you can search.
**3. Example to quote:** > "Ingestion cost: What you pay before you can search."
**Seen in:** main.py, section C+D

## Mock LLM rules  `(library)`
**1. Definition:** MockLLM(rules=[...]) with callables that parse the prompt and return JSON or one word.
**2. In plain words:** Offline stand-in; the same prompts run on a real model.
**3. Example to quote:** > "Mock LLM rules: Offline stand-in; the same prompts run on a real model."
**Seen in:** main.py, MOCK

## extract_json  `(library)`
**1. Definition:** shared.llm_client.extract_json pulls the first {...} object out of a reply; raises ValueError if none.
**2. In plain words:** Models add chatter around JSON.
**3. Example to quote:** > "extract_json: Models add chatter around JSON."
**Seen in:** main.py, proposition_chunks

## Source span of a proposition  `(concept)`
**1. Definition:** The character span of the sentence a proposition came from; used for citation and for scoring.
**2. In plain words:** Without it a rewritten fact cannot be cited.
**3. Example to quote:** > "Source span of a proposition: Without it a rewritten fact cannot be cited."
**Seen in:** main.py, proposition_chunks

## Lexical overlap as topic proxy  `(concept)`
**1. Definition:** Share of the new sentence's content words that already occur in the current chunk.
**2. In plain words:** The mock's definition of 'same topic'.
**3. Example to quote:** > "Lexical overlap as topic proxy: The mock's definition of 'same topic'."
**Seen in:** main.py, _agent_rule
