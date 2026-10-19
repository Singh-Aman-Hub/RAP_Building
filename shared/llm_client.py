"""
shared/llm_client.py  --  ONE tiny text-in / text-out LLM interface used by modules 07 and 08.

Why this file exists
--------------------
Prompt-engineering and agent-harness examples need "an LLM" but you may not
have an API key or a GPU.  So every example talks to the same function:

    reply_text = generate(messages, system="...", max_tokens=300, temperature=0.0)

and the *backend* behind it can be swapped with ONE environment variable:

    LLM_BACKEND=mock        (default)  scripted / rule-based fake, instant, offline, deterministic
    LLM_BACKEND=local                  small open model via Hugging Face transformers (CPU ok, ~1 GB download)
    LLM_BACKEND=anthropic              real hosted model (needs ANTHROPIC_API_KEY)

Message format (the same shape almost every chat API uses):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": "What is 2+2?"},
    ]
    `system` is passed separately (Anthropic style).  The local backend folds it in
    as a first {"role": "system"} message because that is what chat templates expect.

How to run the self-test:
    python shared/llm_client.py
"""

from __future__ import annotations  # lets us use modern type-hint syntax on older Pythons

import json
import os
import re
from typing import Callable

Message = dict  # {"role": "user"|"assistant", "content": str}


# --------------------------------------------------------------------------- #
# Backend 1: MockLLM  (no network, no model, fully deterministic)
# --------------------------------------------------------------------------- #
class MockLLM:
    """A fake LLM.  Two ways to control it:

    1. `script=[...]`  : a list of replies that are returned one by one, in order.
                         Perfect for agent-loop demos: you decide the "model's" next move.
    2. `rules=[(regex, reply_or_callable), ...]` : first regex that matches the LAST user
                         message wins.  A callable receives (messages, system) -> str.

    If nothing matches, it returns a polite fallback string.
    """

    def __init__(self, script: list[str] | None = None,
                 rules: list[tuple[str, str | Callable]] | None = None):
        self.script = list(script) if script else []
        self.rules = rules or []
        self.calls: list[dict] = []  # every request is recorded -> great for debugging/tests

    def generate(self, messages: list[Message], system: str | None = None,
                 max_tokens: int = 300, temperature: float = 0.0) -> str:
        self.calls.append({"messages": messages, "system": system})
        if self.script:                      # scripted mode: pop the next canned reply
            return self.script.pop(0)
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        for pattern, reply in self.rules:    # rule mode: regex on last user message
            if re.search(pattern, last_user, flags=re.IGNORECASE):
                return reply(messages, system) if callable(reply) else reply
        return "I am a mock model. Configure me with script=[...] or rules=[...]."


# --------------------------------------------------------------------------- #
# Backend 2: local open model (Hugging Face transformers)
# --------------------------------------------------------------------------- #
_LOCAL_PIPE = None  # cache so the model loads only once per process


def _generate_local(messages, system, max_tokens, temperature) -> str:
    global _LOCAL_PIPE
    # Imported lazily: transformers/torch are heavy and not needed for the mock backend.
    from transformers import pipeline

    if _LOCAL_PIPE is None:
        model_id = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        _LOCAL_PIPE = pipeline("text-generation", model=model_id)
    chat = ([{"role": "system", "content": system}] if system else []) + list(messages)
    do_sample = temperature > 0
    out = _LOCAL_PIPE(
        chat,
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
    )
    # With chat input, the pipeline returns the whole conversation; the last item is the new reply.
    return out[0]["generated_text"][-1]["content"]


# --------------------------------------------------------------------------- #
# Backend 3: Anthropic API
# --------------------------------------------------------------------------- #
def _generate_anthropic(messages, system, max_tokens, temperature) -> str:
    import anthropic  # lazy import

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    resp = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
        max_tokens=max_tokens,
        temperature=temperature,
        system=system or "",
        messages=messages,
    )
    # resp.content is a list of blocks; text replies are blocks with type == "text"
    return "".join(block.text for block in resp.content if block.type == "text")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
_DEFAULT_MOCK = MockLLM()


def generate(messages: list[Message], system: str | None = None,
             max_tokens: int = 300, temperature: float = 0.0,
             backend: str | None = None, mock: MockLLM | None = None) -> str:
    """Send `messages` to the selected backend and return the reply text.

    backend : "mock" | "local" | "anthropic".  Defaults to $LLM_BACKEND, else "mock".
    mock    : pass your own configured MockLLM (only used when backend == "mock").
    """
    backend = backend or os.getenv("LLM_BACKEND", "mock")
    if backend == "mock":
        return (mock or _DEFAULT_MOCK).generate(messages, system, max_tokens, temperature)
    if backend == "local":
        return _generate_local(messages, system, max_tokens, temperature)
    if backend == "anthropic":
        return _generate_anthropic(messages, system, max_tokens, temperature)
    raise ValueError(f"Unknown LLM backend {backend!r}. Use mock, local or anthropic.")


def extract_json(text: str) -> dict:
    """Pull the first {...} JSON object out of an LLM reply (models often add chatter around it).

    Raises ValueError if no valid JSON object is found -- callers should catch it and retry.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)  # greedy: first '{' to last '}'
    if not match:
        raise ValueError(f"No JSON object found in model reply: {text!r}")
    return json.loads(match.group(0))  # may raise json.JSONDecodeError (a ValueError subclass)


if __name__ == "__main__":
    demo = MockLLM(rules=[(r"2\s*\+\s*2", "4"), (r"hello", "Hi there!")])
    print(generate([{"role": "user", "content": "Hello!"}], mock=demo))
    print(generate([{"role": "user", "content": "What is 2 + 2?"}], mock=demo))
    print(extract_json('Sure! Here you go: {"label": "positive", "score": 0.9} Hope that helps.'))
