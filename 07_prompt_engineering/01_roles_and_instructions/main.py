"""
GOAL   : See how role / task / instruction wording and the system-vs-user split change
         what comes back from the model (zero-shot prompting = no examples given).
RUN    : cd 07_prompt_engineering/01_roles_and_instructions && python main.py
         (from repo root: .venv/bin/python 07_prompt_engineering/01_roles_and_instructions/main.py)
EXPECTED OUTPUT (mock backend = SIMULATION, see sim_llm.py):
         three replies: a chatty one for the vague prompt, a sentence for the half-specified
         prompt, and a bare label for the precise prompt.
NEEDS  : nothing (mock backend). For real results: LLM_BACKEND=local or anthropic.
"""
import sys
from pathlib import Path

# Make sim_llm.py (one folder up) and shared/llm_client.py importable from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_llm import make_sim                      # our simulated model (module-local helper)
from llm_client import generate                   # the shared text-in/text-out function

SIM = make_sim()  # used only when LLM_BACKEND=mock; ignored by real backends

review = "The delivery was slow and the box was broken."

# ---- Prompt A: vague. No role, no labels, no output format. -------------------
vague = [{"role": "user", "content": f"What do you think about this? Text: {review}"}]

# ---- Prompt B: task + labels, but no output constraint. ------------------------
task_only = [{
    "role": "user",
    "content": f"Classify the sentiment as positive, negative or neutral.\nText: {review}",
}]

# ---- Prompt C: ROLE + TASK + INSTRUCTIONS(constraints) split across system/user ---
# system  = stable rules that apply to every request (who you are, what format to use)
# user    = the changing input for THIS request
SYSTEM_C = (
    "You are a sentiment classifier for customer reviews.\n"          # role
    "Allowed labels: positive, negative, neutral.\n"                   # closed set
    "Reply with the label only, in lower case, nothing else."          # output constraint
)
precise = [{"role": "user", "content": f"Text: {review}"}]

for name, messages, system in [
    ("A vague", vague, None),
    ("B task only", task_only, None),
    ("C role+task+constraints (system/user split)", precise, SYSTEM_C),
]:
    # temperature=0.0 -> least random output, best for classification & testing
    reply = generate(messages, system=system, temperature=0.0, mock=SIM)
    print(f"{name:45s} -> {reply!r}")

# TRY IT: delete the sentence "Reply with the label only..." from SYSTEM_C and re-run.
# TRY IT: change `review` to "I love it" and watch all three prompts change.
# BREAK IT: write messages=[{"role": "robot", "content": "hi"}] and run with
#           LLM_BACKEND=anthropic -> the API rejects unknown roles (a 400 error).
