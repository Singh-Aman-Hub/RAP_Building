"""Step 01 exercises: token windows of 64 with overlap 10, and Japanese / Arabic tokenizer ratios.
RUN: HF_HUB_OFFLINE=1 ../../.venv/bin/python s01_windows_and_unk.py"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common19 import count_tokens, get_tokenizer, load_docs, load_step  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

s1 = load_step("01_tokens_not_characters")
hb = next(d for d in load_docs() if d["doc_id"] == "employee_handbook")
w = s1.fixed_token_windows(hb, 64, 10)
print(f"size=64 overlap=10 -> {len(w)} windows, largest {max(count_tokens(c.text) for c in w)} tokens")

samples = {"Japanese": "試用期間の後、従業員は週に最大3日まで在宅勤務ができます。",
           "Arabic": "يمكن للموظفين العمل عن بعد حتى ثلاثة أيام في الأسبوع بعد فترة التجربة."}
toks = {"MiniLM": get_tokenizer(), "e5-small": AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small"),
        "gpt2": AutoTokenizer.from_pretrained("gpt2")}
for lang, s in samples.items():
    row = {n: len(t(s, add_special_tokens=False)["input_ids"]) for n, t in toks.items()}
    unk = sum(i == toks["MiniLM"].unk_token_id for i in toks["MiniLM"](s, add_special_tokens=False)["input_ids"])
    print(f"{lang}: {len(s)} chars -> tokens {row}; MiniLM [UNK] = {unk}")

try:
    s1.fixed_token_windows(hb, 128, 128)
except AssertionError as e:
    print("overlap == size ->", type(e).__name__, "(the guard fires before any loop starts)")
