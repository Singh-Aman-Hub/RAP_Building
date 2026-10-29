# 09: LLM Context and Grounding

**Level:** Intermediate · **Intentional bugs:** 4 · **Dependencies:** standard library only (no API key)

## Scenario
Your company's support bot is accused of two things in production: it **refuses questions it should answer**, and it **confidently invents answers** to questions the documents don't cover. Retrieval has been checked and looks fine. The suspicion is on the code that sits *between* retrieval and the LLM: choosing context, building the prompt, the refusal rule, and citation mapping.

The interviewer gives you that module with a deterministic stand-in for the LLM (`fake_llm`). The stub behaves like a real model: it answers from the sources it is shown, cites the label it saw, and makes something up when nothing relevant is in the prompt. **`fake_llm` is not buggy. Do not edit it.**

## Expected behaviour
For the three test questions, `python buggy_code.py` should print:

| Question | Expected answer | Cited source | Chunks in prompt |
|---|---|---|---|
| Within how many days can items be returned? | the **30 days** sentence (current policy) | `refund_policy_2025.md` | `c1` first; **never** `c2` |
| How long does express delivery take? | the **express delivery / two working days** sentence | `shipping_policy.md` | `c3` first |
| Do you sell gift cards? | exactly the refusal string | none | none (the LLM is not called) |

General rules the module must follow:
- **Refuse** when the best retrieval score is below `REFUSAL_THRESHOLD`. Otherwise answer.
- **Archived** documents must never reach the prompt.
- The prompt contains the **most relevant** chunks that fit into `MAX_CONTEXT_WORDS`.
- A citation `[n]` in the answer maps to the chunk the model saw labelled `[n]`. The system prompt tells the model sources are numbered like `[1]`.

## Observed behaviour
```
Q: Within how many days can items be returned?
A: I don't know based on the available documents.
   cited sources: []
   chunks in prompt: []

Q: How long does express delivery take?
A: I don't know based on the available documents.
   ...

Q: Do you sell gift cards?
A: Yes, absolutely! We offer this and it usually arrives within 24 hours.
   cited sources: []
   chunks in prompt: ['c1', 'c4']
```
The two answerable questions are refused, and the unanswerable one gets a hallucination. Once you fix the first problem, new wrong outputs appear: wrong policy version, a citation pointing at a document that doesn't contain the quoted sentence, and more. Keep going until the table above matches exactly.

## Your task
Fix the 4 bugs in the code under test (`select_context`, `build_prompt`, `parse_citations`, `answer_question`). Record each one in the README bug log.

## Interview follow-ups
- Why is a wrong citation arguably worse than no citation?
- Where would you choose `REFUSAL_THRESHOLD` from in a real system, and what does it trade off?
- If two policy versions are both retrieved, which layer should prevent the old one from being used: the index, the retriever or the prompt? Why?
