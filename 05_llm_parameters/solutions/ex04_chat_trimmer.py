"""Solution ex04: trim a chat history to a token budget, always keeping the system message.
This is the same idea the agent harness (module 08) needs.
RUN: .venv/bin/python 05_llm_parameters/solutions/ex04_chat_trimmer.py"""
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("distilgpt2")


def n_tokens(message):
    return len(tok.encode(message["content"]))


def trim_history(messages, budget):
    """messages[0] is the system prompt (kept). Drop the OLDEST other messages until we fit."""
    system, rest = messages[0], messages[1:]
    kept, used = [], n_tokens(system)
    for msg in reversed(rest):                            # walk from newest to oldest
        cost = n_tokens(msg)
        if used + cost > budget:
            break                                         # everything older than this is dropped
        kept.append(msg)
        used += cost
    return [system] + list(reversed(kept)), used          # restore chronological order


history = [{"role": "system", "content": "You are a helpful assistant."}] + [
    {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message number {i}: " + "word " * 20}
    for i in range(10)
]
trimmed, used = trim_history(history, budget=120)
print(f"kept {len(trimmed)} of {len(history)} messages using {used} tokens")
print("oldest kept:", trimmed[1]["content"][:20])
assert trimmed[0]["role"] == "system" and used <= 120
