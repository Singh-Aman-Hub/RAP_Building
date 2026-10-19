# Terms — Step 05: Memory (short-term vs long-term)

Folder: `08_agent_harness/05_memory` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [08 Agent harness README](../README.md)

**Terms in this folder:** [Short-term memory](#short-term-memory) · [Long-term memory](#long-term-memory) · [Memory tools (remember, recall, list_memories)](#memory-tools-remember-recall-list_memories) · [Key-value store](#key-value-store) · [pathlib.Path and the / operator](#pathlibpath-and-the--operator) · [Path file methods (exists, read_text, write_text, mkdir, unlink)](#path-file-methods-exists-read_text-write_text-mkdir-unlink) · [json.loads and json.dumps](#jsonloads-and-jsondumps) · [dict.get with a default](#dictget-with-a-default) · [Slicing a string to cap length ([:N])](#slicing-a-string-to-cap-length-n) · [or for a default value](#or-for-a-default-value) · [Tool availability (allow-list per run)](#tool-availability-allow-list-per-run) · [Conditional expression for dispatch](#conditional-expression-for-dispatch) · [Keyword unpacking (**decision["args"])](#keyword-unpacking-decisionargs) · [Memory poisoning](#memory-poisoning) · [Memory design questions (size, secrets, concurrency)](#memory-design-questions-size-secrets-concurrency) · [if __name__ == "__main__"](#if-__name__--__main__) · [_ as a throwaway loop variable](#_-as-a-throwaway-loop-variable) · [FileNotFoundError](#filenotfounderror)

---

## Short-term memory
*Type: concept*

**1. Definition:** The `messages` list of the current agent run; it disappears when the run ends.
**2. In plain words:** It is what the model can "see" right now. A brand-new run starts with an empty list, so nothing from yesterday is known unless you provide it.
**3. Example to quote:**
> "Session 2 starts with a fresh `messages` list containing only the question, so the agent has no idea the name is Priya unless it reads memory."

**Seen in:** `main.py`, `run_agent` (`messages = [...]`).

## Long-term memory
*Type: concept*

**1. Definition:** Information stored outside the model and the current run, here a JSON file, that survives between runs and is accessed through tools.
**2. In plain words:** The model itself remembers nothing, so persistence has to be built by you. In production this is a database or a vector store.
**3. Example to quote:**
> "Session 1 stored the name with a tool, and session 2 recalled it from `memory.json` even though the message list was new."

**Seen in:** `main.py`, `MEMORY_FILE` and the memory tools.

## Memory tools (`remember`, `recall`, `list_memories`)
*Type: project functions*

**1. Definition:** Three tools that write, read and list entries in the long-term store.
**2. In plain words:** The model asks for them by name in JSON, and your code touches the file. The control run removes the tools to prove that the file, not the model, carries the memory.
**3. Example to quote:**
> "`remember('name', 'Priya')` returns `stored 'name'`, and `recall('name')` later returns `Priya`."

**Seen in:** `main.py`, `remember`, `recall`, `list_memories`, `TOOLS_WITH_MEMORY`.

## Key-value store
*Type: concept*

**1. Definition:** A data structure that maps each key to a value, here a dictionary persisted as JSON.
**2. In plain words:** Lookup is exact: you must ask for the same key you stored. Finding "similar" notes needs embeddings instead (module 04).
**3. Example to quote:**
> "`memory.json` contains `{\"name\": \"Priya\"}`; recalling `name` works, recalling `nickname` returns 'nothing stored'."

**Seen in:** `main.py`, `load_memory`, `save_memory`.

## pathlib.Path and the `/` operator
*Type: Python standard library*

**1. Definition:** `Path` represents a filesystem path; `Path / "data" / "memory.json"` joins path pieces portably.
**2. In plain words:** Building the file path from the script's folder keeps it working from any current directory and on any operating system.
**3. Example to quote:**
> "`MEMORY_FILE = HERE / 'data' / 'memory.json'` puts the file in a `data` folder next to the script."

**Seen in:** `main.py`, `MEMORY_FILE`.

## Path file methods (`exists`, `read_text`, `write_text`, `mkdir`, `unlink`)
*Type: Python standard library*

**1. Definition:** `exists()` tests presence, `read_text()` and `write_text(s)` read and write whole text files, `mkdir(parents=True, exist_ok=True)` creates folders, `unlink(missing_ok=True)` deletes a file without erroring if absent.
**2. In plain words:** These cover everything the memory store needs. Forgetting `mkdir` on a new machine causes `FileNotFoundError` when saving.
**3. Example to quote:**
> "`MEMORY_FILE.unlink(missing_ok=True)` starts the demo from a clean slate every time."

**Seen in:** `main.py`, `load_memory`, `save_memory`, `__main__` block.

## json.loads and json.dumps
*Type: Python standard library*

**1. Definition:** `json.loads(text)` parses JSON text into Python objects; `json.dumps(obj, indent=2)` converts objects to formatted JSON text.
**2. In plain words:** They convert between the dict you use in code and the text stored on disk. `indent=2` makes the file readable.
**3. Example to quote:**
> "`save_memory` writes `json.dumps(memory, indent=2)` and `load_memory` reads it back with `json.loads`."

**Seen in:** `main.py`, `load_memory`, `save_memory`.

## dict.get with a default
*Type: Python built-in*

**1. Definition:** `d.get(key, default)` returns the value for `key` or `default` if the key is missing, without raising `KeyError`.
**2. In plain words:** It lets `recall` answer politely when nothing is stored. The same method on the tools dict returns `None` for an unknown tool, which the loop reports as an error.
**3. Example to quote:**
> "`memory.get(key, f'nothing stored for {key!r}')` returns a friendly message for a missing key."

```python
{}.get("a", 0)   # 0
```

**Seen in:** `main.py`, `recall`, `run_agent`.

## Slicing a string to cap length (`[:N]`)
*Type: Python syntax*

**1. Definition:** `text[:200]` returns the first 200 characters of a string.
**2. In plain words:** It stops the model from storing unlimited text. Silent truncation can surprise users, so decide whether to warn.
**3. Example to quote:**
> "With `MAX_VALUE_CHARS = 3` the name Priya is stored as `Pri`."

```python
"hello"[:2]   # 'he'
```

**Seen in:** `main.py`, `remember`.

## `or` for a default value
*Type: Python syntax*

**1. Definition:** `a or b` returns `a` if it is truthy, otherwise `b`; an empty string is falsy.
**2. In plain words:** `", ".join(load_memory()) or "(empty)"` shows `(empty)` when there are no keys.
**3. Example to quote:**
> "`list_memories()` prints `(empty)` instead of a blank string when nothing is stored."

```python
"" or "x"   # 'x'
```

**Seen in:** `main.py`, `list_memories`.

## Tool availability (allow-list per run)
*Type: concept*

**1. Definition:** Passing a different `tools` dictionary to each run to control which tools the agent may use (`TOOLS_WITH_MEMORY` vs an empty dict).
**2. In plain words:** A tool that is not in the dict cannot run, and asking for it returns an error observation. This is also a safety feature.
**3. Example to quote:**
> "In the control run `recall` is not available, so the answer is 'ERROR: tool 'recall' is not available'."

**Seen in:** `main.py`, `TOOLS_WITHOUT_MEMORY`, `run_agent`.

## Conditional expression for dispatch
*Type: Python syntax*

**1. Definition:** `fn(**args) if fn else "ERROR..."` calls the function when it exists and otherwise returns an error string.
**2. In plain words:** It turns "unknown tool" into data for the model instead of a crash.
**3. Example to quote:**
> "`result = fn(**decision['args']) if fn else f'ERROR: tool {name!r} is not available'`."

**Seen in:** `main.py`, `run_agent`.

## Keyword unpacking (`**decision["args"]`)
*Type: Python syntax*

**1. Definition:** Passing a dict's items as keyword arguments to a function.
**2. In plain words:** The model's `args` become named parameters of the tool function.
**3. Example to quote:**
> "`remember(**{'key': 'name', 'value': 'Priya'})` is the same as `remember(key='name', value='Priya')`."

**Seen in:** `main.py`, `run_agent`.

## Memory poisoning
*Type: concept (security)*

**1. Definition:** Storing malicious instructions in long-term memory (for example text copied from an untrusted web page) so that they influence the agent in later sessions.
**2. In plain words:** What you store today becomes part of tomorrow's prompt. Treat memory contents as untrusted unless you wrote them.
**3. Example to quote:**
> "If the agent saves text from a web page that says 'ignore your rules', every future session could read that instruction."

**Seen in:** `main.py`, closing comments.

## Memory design questions (size, secrets, concurrency)
*Type: concept*

**1. Definition:** Points to decide when designing memory: what may be stored (never secrets), how large it may grow (cap and expire), how to retrieve (exact key vs embeddings), and how to handle multiple writers (a database instead of one JSON file).
**2. In plain words:** The demo is simple on purpose; interviewers ask about these limits.
**3. Example to quote:**
> "Two processes writing one JSON file can corrupt it, so for real use I would switch to a database."

**Seen in:** `main.py`, closing comments.

## if __name__ == "__main__"
*Type: Python syntax*

**1. Definition:** A guard that runs the code under it only when the file is run directly, not when imported.
**2. In plain words:** Other steps can import `run_agent` and the tools without triggering the demo or deleting the memory file.
**3. Example to quote:**
> "The demo that deletes `memory.json` sits under the guard so importing this file is safe."

**Seen in:** `main.py`, bottom section.

## `_` as a throwaway loop variable
*Type: Python syntax*

**1. Definition:** By convention `_` names a loop variable whose value is not used, as in `for _ in range(max_steps)`.
**2. In plain words:** It tells readers the loop only repeats a fixed number of times.
**3. Example to quote:**
> "`for _ in range(max_steps)` runs at most five rounds and I never need the counter."

**Seen in:** `main.py`, `run_agent`.

## FileNotFoundError
*Type: Python built-in exception*

**1. Definition:** The exception raised when opening or writing a file whose path (or parent folder) does not exist.
**2. In plain words:** The traceback shows the missing path. Here it appears if `data/` is missing and the `mkdir` line was removed.
**3. Example to quote:**
> "I deleted the `mkdir` line and the `data` folder and got `FileNotFoundError` pointing at `data/memory.json`."

**Seen in:** `main.py` BREAK IT comment; `WALKTHROUGH.md` debugging practice.
