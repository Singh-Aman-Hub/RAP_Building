### Token (LLM)
<!-- cat: llm -->
- **Technical definition:** The unit of text a language model reads and writes: a sub-word piece (often a word fragment, whole word or punctuation mark) mapped to an integer id by the tokenizer's vocabulary.
- **Plain English:** Models do not see letters or words; they see numbered chunks. "unbelievable" might become 3 tokens. Prompt length, cost, speed and the context limit are all counted in tokens, not words.
- **Example / analogy:** Like Lego bricks of different sizes: common words are one big brick, rare words are built from several small ones.
- **Snippet:**
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("distilgpt2")
print(tok.encode("Hello world"))   # a short list of integer ids
```
- **Snippet explained:** `encode` turns the string into token ids; the length of the list is the token count.
- **Related terms:** [[Prompt]], [[Context window]], [[Tokenizer]]

### Label bias
<!-- cat: llm -->
- **Technical definition:** A tendency of a model, in few-shot prompting, to favour certain labels because of how often they appear, or where they appear, among the examples.
- **Plain English:** If most of your examples say "positive", the model leans toward answering "positive" even when the text is not.
- **Example / analogy:** A quiz where the last five answers were all "B" makes you guess "B" next.
- **Snippet:** n/a
- **Snippet explained:** Mitigate by balancing and shuffling examples across labels.
- **Related terms:** [[Few-shot prompting]]

### JSONL
<!-- cat: python -->
- **Technical definition:** "JSON Lines": a text format with one complete JSON object per line, so files can be appended to and read line by line.
- **Plain English:** A log file where each line is a self-contained record you can parse with `json.loads`; one broken line does not ruin the rest.
- **Example / analogy:** A notebook where every line is a separate, complete index card.
- **Snippet:**
```python
import json
with open("trace.jsonl", "a") as f:
    f.write(json.dumps({"tool": "calc", "ok": True}) + "\n")
```
- **Snippet explained:** `json.dumps` makes one line of JSON; the newline separates records; mode `"a"` appends.
- **Related terms:** [[JSON (JavaScript Object Notation)]]

### Exception handling
<!-- cat: python -->
- **Technical definition:** Using `try` / `except` / `else` / `finally` to catch errors raised at run time and respond instead of letting the program crash.
- **Plain English:** You say "try this; if it fails in this specific way, do that instead". Catch only the errors you can actually handle.
- **Example / analogy:** A safety net under a trapeze act.
- **Snippet:**
```python
try:
    value = int("abc")
except ValueError as err:
    print("not a number:", err)
```
- **Snippet explained:** `int("abc")` raises `ValueError`; the `except` block runs and the program continues.
- **Related terms:** [[ValidationError]]

### Delimiters
<!-- cat: llm -->
- **Technical definition:** Marker strings (such as triple quotes, XML-style tags or `###`) that visibly separate instructions from untrusted data inside a prompt.
- **Plain English:** They tell the model "everything between these markers is data, not commands", which makes prompt injection harder (not impossible).
- **Example / analogy:** Putting a customer's letter in a sealed envelope before handing it to your assistant.
- **Snippet:**
```python
prompt = f"Summarise the text between <doc> tags.\n<doc>\n{user_text}\n</doc>"
```
- **Snippet explained:** The f-string wraps the untrusted `user_text` in tags the instructions refer to.
- **Related terms:** [[Prompt injection]]

### Regular expression
<!-- cat: python -->
- **Technical definition:** A compact pattern language for matching and extracting text, available in Python through the `re` module.
- **Plain English:** A search pattern more powerful than "contains": for example "the first {...} block in this reply".
- **Example / analogy:** Find-and-replace on steroids.
- **Snippet:**
```python
import re
re.search(r"\d+", "order 66 shipped").group()   # '66'
```
- **Snippet explained:** `\d+` means one or more digits; `search` finds the first match and `.group()` returns its text.
- **Related terms:** [[Parser (reply parser)]]

### GIL
<!-- cat: python -->
- **Technical definition:** The Global Interpreter Lock: a lock in standard CPython that lets only one thread execute Python bytecode at a time.
- **Plain English:** Threads do not speed up CPU-heavy pure-Python code, though they help while waiting on network or disk. For CPU-bound work use multiple processes (as Gunicorn workers do) or libraries that release the lock.
- **Example / analogy:** One microphone in a room: many people can wait, only one speaks at once.
- **Snippet:** n/a
- **Snippet explained:** n/a
- **Related terms:** [[Concurrency]]

### Config
<!-- cat: api -->
- **Technical definition:** The settings an application reads at start-up (ports, file paths, model names, limits), kept outside the code in files or environment variables.
- **Plain English:** Change behaviour without editing code: the same app runs on your laptop and in a container with different config.
- **Example / analogy:** The settings menu of a game versus its source code.
- **Snippet:**
```python
import os
PORT = int(os.getenv("PORT", "8000"))
```
- **Snippet explained:** Read `PORT` from the environment, defaulting to 8000.
- **Related terms:** [[Environment variable]]

### Model loading
<!-- cat: llm -->
- **Technical definition:** Reading trained weights from disk (or the network) into memory and constructing the model object so it can make predictions.
- **Plain English:** Slow and memory-hungry, so do it once at start-up, not on every request.
- **Example / analogy:** Starting a car engine once, rather than before every metre.
- **Snippet:**
```python
import joblib
model = joblib.load("model.joblib")   # do this at start-up
```
- **Snippet explained:** `joblib.load` deserialises the saved model into memory.
- **Related terms:** [[Lifespan (startup/shutdown)]], [[Lazy loading]]

### Docker ENV
<!-- cat: api -->
- **Technical definition:** The Dockerfile instruction `ENV KEY=value` that sets an environment variable inside the image and every container started from it.
- **Plain English:** Gives containers their default configuration; `docker run -e` can override it at launch.
- **Example / analogy:** Factory default settings you can still change when you switch the device on.
- **Snippet:**
```dockerfile
ENV MODEL_PATH=/app/model.joblib
```
- **Snippet explained:** The app reads `MODEL_PATH` with `os.getenv`.
- **Related terms:** [[Environment variable]]

### Fixture
<!-- cat: python -->
- **Technical definition:** In pytest, a function marked `@pytest.fixture` that prepares something tests need (a client, a temp file) and is injected by naming it as a test argument.
- **Plain English:** Shared set-up written once and reused by many tests.
- **Example / analogy:** Laying out the tools before each experiment.
- **Snippet:**
```python
import pytest
@pytest.fixture
def numbers():
    return [1, 2, 3]
def test_sum(numbers):
    assert sum(numbers) == 6
```
- **Snippet explained:** pytest sees the argument name `numbers` and passes in the fixture's return value.
- **Related terms:** [[pytest]]

### Docker health check
<!-- cat: api -->
- **Technical definition:** A `HEALTHCHECK` (or Compose `healthcheck`) command Docker runs periodically; the container is marked healthy or unhealthy by its exit code.
- **Plain English:** "Running" is not "working". A health check lets Docker and Compose know when the app is truly ready.
- **Example / analogy:** A nurse taking your pulse every few seconds.
- **Snippet:**
```dockerfile
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
```
- **Snippet explained:** A failing request returns non-zero, so Docker marks the container unhealthy.
- **Related terms:** [[Health check]]

### Host
<!-- cat: api -->
- **Technical definition:** The machine, or the network address of it, on which a server runs; `127.0.0.1` means "this machine only" and `0.0.0.0` means "all network interfaces".
- **Plain English:** Binding to `127.0.0.1` inside a container makes the app unreachable from outside; use `0.0.0.0`.
- **Example / analogy:** A shop's address: "inside the building only" versus "open to the street".
- **Snippet:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
- **Snippet explained:** `--host 0.0.0.0` listens on every interface so Docker's port mapping can reach it.
- **Related terms:** [[Port]], [[Docker port mapping]]

### Docker port mapping
<!-- cat: api -->
- **Technical definition:** The `-p HOST:CONTAINER` option (or Compose `ports`) that forwards a port on the host machine to a port inside the container.
- **Plain English:** Containers are isolated; mapping a port is how your browser reaches the app inside.
- **Example / analogy:** A receptionist forwarding calls from the main number to an extension.
- **Snippet:**
```bash
docker run -p 8080:8000 myapi    # browse http://localhost:8080
```
- **Snippet explained:** Host port 8080 is forwarded to container port 8000.
- **Related terms:** [[Port]], [[Host]]

### Alias
<!-- cat: ml -->
- **Technical definition:** A movable name (such as `latest` or `production`) that points to a specific model version in a registry.
- **Plain English:** Callers ask for "production" and you decide which version that means; promoting or rolling back is just moving the alias.
- **Example / analogy:** A contact named "Mum" pointing to a phone number you can update.
- **Snippet:**
```python
aliases = {"production": "v3"}
version = aliases["production"]
```
- **Snippet explained:** Changing the dictionary value re-points "production" without changing callers.
- **Related terms:** [[Model registry]], [[Model versioning]]
