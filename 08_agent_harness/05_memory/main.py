"""
GOAL   : Give the agent MEMORY. Short-term = the `messages` list of one run. Long-term = a JSON file
         that survives between runs, accessed through tools (remember / recall / list_memories).
RUN    : cd 08_agent_harness/05_memory && python main.py
EXPECTED OUTPUT: session 1 stores the user's name; session 2 (brand-new history) recalls it from disk;
         a control run without the memory tool cannot. memory.json is printed at the end.
NEEDS  : nothing (mock).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
sys.path.insert(0, str(HERE.parent))
from llm_client import generate, extract_json
from fake_llm import make_fake_agent_llm

LLM = make_fake_agent_llm()
MEMORY_FILE = HERE / "data" / "memory.json"        # Path objects: `/` joins path pieces
MAX_VALUE_CHARS = 200                              # never let the model store unlimited text

# ---------------------------------------------------------------------------
# Long-term memory = key/value store on disk. (Real systems: database / vector store - see module 10.)
# ---------------------------------------------------------------------------
def load_memory() -> dict:
    if not MEMORY_FILE.exists():
        return {}
    return json.loads(MEMORY_FILE.read_text())     # file text -> dict

def save_memory(memory: dict) -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)         # create data/ if missing
    MEMORY_FILE.write_text(json.dumps(memory, indent=2))          # dict -> pretty JSON text

def remember(key: str, value: str) -> str:
    memory = load_memory()
    memory[key] = str(value)[:MAX_VALUE_CHARS]     # slice = keep only the first N characters
    save_memory(memory)
    return f"stored {key!r}"

def recall(key: str) -> str:
    memory = load_memory()
    return memory.get(key, f"nothing stored for {key!r}")        # dict.get(key, default)

def list_memories() -> str:
    return ", ".join(load_memory()) or "(empty)"

TOOLS_WITH_MEMORY = {"remember": remember, "recall": recall, "list_memories": list_memories}
TOOLS_WITHOUT_MEMORY = {}

SYSTEM_PROMPT = """You are an assistant with tools. Reply with ONLY JSON:
{"tool": "remember", "args": {"key": ..., "value": ...}} | {"tool": "recall", "args": {"key": ...}} | {"final": "..."}"""

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

if __name__ == "__main__":
    MEMORY_FILE.unlink(missing_ok=True)            # start the demo from a clean slate (repeatable)

    print("SESSION 1")
    print("  user : Remember that my name is Priya")
    print("  agent:", run_agent("Remember that my name is Priya", TOOLS_WITH_MEMORY))

    print("\nSESSION 2 (new process, new empty message list)")
    print("  user : What is my name?")
    print("  agent:", run_agent("What is my name?", TOOLS_WITH_MEMORY))

    print("\nCONTROL: same question, but the agent has no memory tool")
    print("  agent:", run_agent("What is my name?", TOOLS_WITHOUT_MEMORY))

    print("\nmemory.json on disk:\n" + MEMORY_FILE.read_text())

# Limitations / design questions you should be able to discuss:
#  * What may be stored? (never secrets/passwords)  * How big may it grow? (cap size, expire old keys)
#  * Memory POISONING: text stored from an untrusted web page can later inject instructions (module 07/06).
#  * Retrieval: key lookup is exact. Real long-term memory often uses embeddings (module 04) to find
#    *similar* notes.  * Concurrency: two processes writing one JSON file can corrupt it -> use a database.
# TRY IT: change MAX_VALUE_CHARS to 3 and store the name again. TRY IT: ask for a key that does not exist.
# BREAK IT: delete the mkdir line and remove the data/ folder: read the FileNotFoundError traceback.
