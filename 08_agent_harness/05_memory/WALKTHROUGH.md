# Step 05 — Memory: short-term vs long-term

## The story so far
In [step 04](../04_history_and_context/) history stays inside the budget, but it dies when the program ends and trimming loses old facts.

## The pain point
Short-term memory (the message list) is temporary and capped. An assistant that forgets your preferences every run is not useful.

## The spark
Long-term memory is just storage the agent reaches through tools: `remember(key, value)` and `recall(key)` backed by a JSON file. The model decides what to save; your code persists it.

## Goal
Give the agent facts that survive between runs using a JSON file and `remember` / `recall` tools.

## Run it
```bash
cd 08_agent_harness/05_memory && ../../.venv/bin/python main.py
```
Real output:
```
SESSION 1
  user : Remember that my name is Priya
      [remember] -> stored 'name'
  agent: Result: stored 'name'

SESSION 2 (new process, new empty message list)
  user : What is my name?
      [recall] -> Priya
  agent: Result: Priya

CONTROL: same question, but the agent has no memory tool
      [recall] -> ERROR: tool 'recall' is not available
  agent: Sorry, the tool failed: ERROR: tool 'recall' is not available

memory.json on disk:
{
  "name": "Priya"
}
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Short-term memory | working memory, context, message list | Same | Lives in the `messages` list for one run only. |
| Long-term memory | persistent memory, memory store, agent memory | Same | Saved outside the model (file/DB/vector store) and survives runs. |
| Key-value store | dict, JSON file, KV memory | Near-same | Exact-key lookup; vector memory finds similar text instead. |
| Memory poisoning | stored prompt injection | Related | Malicious text saved today steers the agent tomorrow. |
| Session | run, conversation, thread | Near-same | One execution with one fresh message list. |

## Code, step by step

### 1. Imports and path setup  (`main.py` lines 9–17)

```python
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
sys.path.insert(0, str(HERE.parent))
from llm_client import generate, extract_json
from fake_llm import make_fake_agent_llm
```

- **In simple words:** Load JSON, paths, the LLM helper and the scripted model.
- **Exactly:** `json` converts between dict and text; the rest is the same setup as earlier steps.
- **Why it is here:** Memory needs file I/O and JSON.

### 2. Brain, memory file path, size cap  (`main.py` lines 19–21)

```python
LLM = make_fake_agent_llm()
MEMORY_FILE = HERE / "data" / "memory.json"        # Path objects: `/` joins path pieces
MAX_VALUE_CHARS = 200                              # never let the model store unlimited text
```

- **In simple words:** Choose where memory is stored and how much may be saved.
- **Exactly:** `MEMORY_FILE = HERE / "data" / "memory.json"` (`Path.__truediv__` joins paths). `MAX_VALUE_CHARS = 200`.
- **Why it is here:** The cap stops a model from filling the disk or context with unlimited text.

### 3. Load memory from disk  (`main.py` lines 26–29)

```python
def load_memory() -> dict:
    if not MEMORY_FILE.exists():
        return {}
    return json.loads(MEMORY_FILE.read_text())     # file text -> dict
```

- **In simple words:** Read the saved notes, or start empty.
- **Exactly:** Returns `{}` if the file is missing; else `json.loads(text)` -> `dict`.
- **Why it is here:** First run has no file; without the check it would raise `FileNotFoundError`.

### 4. Save memory to disk  (`main.py` lines 31–33)

```python
def save_memory(memory: dict) -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)         # create data/ if missing
    MEMORY_FILE.write_text(json.dumps(memory, indent=2))          # dict -> pretty JSON text
```

- **In simple words:** Write the dict as pretty JSON.
- **Exactly:** `mkdir(parents=True, exist_ok=True)` creates `data/`; `json.dumps(memory, indent=2)` makes indented text.
- **Why it is here:** Without `mkdir` the first write fails if `data/` is absent.

### 5. The remember tool  (`main.py` lines 35–39)

```python
def remember(key: str, value: str) -> str:
    memory = load_memory()
    memory[key] = str(value)[:MAX_VALUE_CHARS]     # slice = keep only the first N characters
    save_memory(memory)
    return f"stored {key!r}"
```

- **In simple words:** Store one fact.
- **Exactly:** Load, set `memory[key] = str(value)[:200]` (slice cuts to 200 characters), save, return `stored 'name'` (real output).
- **Why it is here:** Read-modify-write keeps old keys; `str()` forces text.

### 6. The recall tool  (`main.py` lines 41–43)

```python
def recall(key: str) -> str:
    memory = load_memory()
    return memory.get(key, f"nothing stored for {key!r}")        # dict.get(key, default)
```

- **In simple words:** Look up one fact.
- **Exactly:** `dict.get(key, default)` returns the value, e.g. `Priya`, or `nothing stored for 'x'`.
- **Why it is here:** A missing key must not crash the tool.

### 7. The list_memories tool  (`main.py` lines 45–46)

```python
def list_memories() -> str:
    return ", ".join(load_memory()) or "(empty)"
```

- **In simple words:** List which keys are stored.
- **Exactly:** `", ".join(dict)` joins the keys; `or "(empty)"` is used when the string is empty.
- **Why it is here:** Lets the model see what it knows.

### 8. Two tool sets  (`main.py` lines 48–49)

```python
TOOLS_WITH_MEMORY = {"remember": remember, "recall": recall, "list_memories": list_memories}
TOOLS_WITHOUT_MEMORY = {}
```

- **In simple words:** One agent with memory tools, one without.
- **Exactly:** Dicts name -> function; `TOOLS_WITHOUT_MEMORY = {}` for the control run.
- **Why it is here:** The control proves the answer came from the memory tool.

### 9. System prompt for the protocol  (`main.py` lines 51–52)

```python
SYSTEM_PROMPT = """You are an assistant with tools. Reply with ONLY JSON:
{"tool": "remember", "args": {"key": ..., "value": ...}} | {"tool": "recall", "args": {"key": ...}} | {"final": "..."}"""
```

- **In simple words:** Tell the model the JSON format.
- **Exactly:** Describes `remember`, `recall` and `final` replies.
- **Why it is here:** The harness parses the reply as JSON.

### 10. The agent loop with tool lookup  (`main.py` lines 54–66)

```python
def run_agent(question: str, tools: dict, max_steps: int = 5) -> str:
    messages = [{"role": "user", "content": question}]      # SHORT-TERM memory: fresh list for every run
    for _ in range(max_steps):
        reply = generate(messages, system=SYSTEM_PROMPT, mock=LLM)
        decision = extract_json(reply)
        messages.append({"role": "assistant", "content": reply})
        if "final" in decision:
            return decision["final"]
        fn = tools.get(decision["tool"])
        result = fn(**decision["args"]) if fn else f"ERROR: tool {decision['tool']!r} is not available"
        print(f"      [{decision['tool']}] -> {result}")
        messages.append({"role": "user", "content": f"Observation: {result}"})
    return "STOPPED: max_steps"
```

- **In simple words:** The same loop as step 02, but tools are passed in and missing tools give an error text.
- **Exactly:** `messages` is created fresh per call (short-term memory). `tools.get(name)` returns `None` if absent, so the result becomes `ERROR: tool 'recall' is not available` (real control output). Loop bounded by `range(max_steps)`.
- **Why it is here:** A fresh list per run shows that only the file carries knowledge between sessions.

### 11. Main guard and clean slate  (`main.py` lines 68–69)

```python
if __name__ == "__main__":
    MEMORY_FILE.unlink(missing_ok=True)            # start the demo from a clean slate (repeatable)
```

- **In simple words:** Delete old memory so the demo repeats.
- **Exactly:** `unlink(missing_ok=True)` removes the file without error if absent.
- **Why it is here:** Otherwise a stale file would make the demo irrelevant.

### 12. Session 1: store the name  (`main.py` lines 71–73)

```python
    print("SESSION 1")
    print("  user : Remember that my name is Priya")
    print("  agent:", run_agent("Remember that my name is Priya", TOOLS_WITH_MEMORY))
```

- **In simple words:** Tell the agent your name.
- **Exactly:** Real: `[remember] -> stored 'name'` and `agent: Result: stored 'name'`.
- **Why it is here:** Writes `data/memory.json`.

### 13. Session 2: recall from disk  (`main.py` lines 75–77)

```python
    print("\nSESSION 2 (new process, new empty message list)")
    print("  user : What is my name?")
    print("  agent:", run_agent("What is my name?", TOOLS_WITH_MEMORY))
```

- **In simple words:** A new run with an empty message list asks the name.
- **Exactly:** Real: `[recall] -> Priya`, `agent: Result: Priya`.
- **Why it is here:** Proves persistence across runs.

### 14. Control: no memory tool  (`main.py` lines 79–80)

```python
    print("\nCONTROL: same question, but the agent has no memory tool")
    print("  agent:", run_agent("What is my name?", TOOLS_WITHOUT_MEMORY))
```

- **In simple words:** Same question, no tool.
- **Exactly:** Real: `[recall] -> ERROR: tool 'recall' is not available`, then `Sorry, the tool failed: ...`.
- **Why it is here:** Shows the model alone cannot know the name.

### 15. Print the file  (`main.py` lines 82–82)

```python
    print("\nmemory.json on disk:\n" + MEMORY_FILE.read_text())
```

- **In simple words:** Show what is stored.
- **Exactly:** Prints `{ "name": "Priya" }` (pretty JSON).
- **Why it is here:** Memory is just a visible file.

### 16. Limitations and experiments  (`main.py` lines 84–90)

```python
# Limitations / design questions you should be able to discuss:
#  * What may be stored? (never secrets/passwords)  * How big may it grow? (cap size, expire old keys)
#  * Memory POISONING: text stored from an untrusted web page can later inject instructions (module 07/06).
#  * Retrieval: key lookup is exact. Real long-term memory often uses embeddings (module 04) to find
#    *similar* notes.  * Concurrency: two processes writing one JSON file can corrupt it -> use a database.
# TRY IT: change MAX_VALUE_CHARS to 3 and store the name again. TRY IT: ask for a key that does not exist.
# BREAK IT: delete the mkdir line and remove the data/ folder: read the FileNotFoundError traceback.
```

- **In simple words:** Design questions in comments.
- **Exactly:** Comments only: secrets, size, poisoning, embeddings, concurrency.
- **Why it is here:** These are the usual interview follow-ups.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `Path(...) / "data" / "memory.json"` | build a path with `/` | `Path("a") / "b"` -> `a/b` |
| `path.exists()`, `read_text()`, `write_text()`, `unlink(missing_ok=True)`, `mkdir(parents=True, exist_ok=True)` | pathlib file operations | see file |
| `json.loads(text)` / `json.dumps(obj, indent=2)` | JSON text <-> dict | – |
| `str(value)[:200]` | slice: first 200 characters | `"hello"[:2]` -> `he` |
| `memory.get(key, default)` | read a key with a fallback | `{}.get("a", 0)` -> 0 |
| `", ".join(dict) or "(empty)"` | `or` picks the right side when the left is empty | `"" or "x"` -> `x` |
| `if __name__ == "__main__":` | run the demo only when executed directly (not when imported) | – |

## Libraries & functions
| Name | Kind | Does | Returns |
|---|---|---|---|
| `pathlib.Path` | standard library | object-oriented file paths | Path |
| `json.loads/dumps` | standard library | parse/produce JSON | dict / str |
| `dict.get` | built-in | safe lookup | value or default |
| `tools.get(name)` | built-in | returns None for unknown tools -> we report an error | – |

## Execution trace
Session 1: model asks `remember(name, Priya)` -> file written. Session 2: brand-new `messages`; model asks `recall(name)`; tool reads the file -> `Priya`. Control: no memory tool -> error observation. The only thing that carried over is the file.

## Experiments
- `MAX_VALUE_CHARS = 3` -> stored value becomes `Pri`: limits protect you, but silently truncating may surprise users.
- Recall a missing key -> `nothing stored for 'x'`.

## Debugging practice
Delete `data/` then store: if `FileNotFoundError` appears, the `mkdir` line is missing. Read the path in the error message.

## Rebuild checklist
- [ ] load/save helpers. [ ] 3 memory tools. [ ] fresh `messages` per run.

## The new problem (cliffhanger)
Real tools fail: the network times out, the model returns broken JSON, a tool raises. One failure in a long loop should not crash the agent, and endless retries should not burn money. Next: [06_errors_and_retries](../06_errors_and_retries/).

> **Say it aloud (interview box):** Short-term memory is the message list, which is capped and lost when the program ends. Long-term memory is persistent storage, here a JSON file, exposed to the model as remember and recall tools. The model chooses what to store, and my code is responsible for persisting and validating it.
