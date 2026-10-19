# Step 05 — Memory: short-term vs long-term

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

## Say it aloud
"Short-term memory is the message list of the current run; long-term memory is persisted outside the model — here a JSON file, in production a database or vector store — accessed through tools. I cap size, never store secrets, and remember that stored text from untrusted sources can inject instructions later."
