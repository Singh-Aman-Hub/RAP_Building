"""Solution ex01: let command-line flags override config.json (how real tools behave).
RUN: .venv/bin/python 05_llm_parameters/solutions/ex01_cli_override.py --temperature 0.2 --top_k 10"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_config_file"))
from config_loader import load_config, generation_kwargs

parser = argparse.ArgumentParser(description="Show generation settings after CLI overrides")
parser.add_argument("--temperature", type=float)          # type=float converts the text '0.2' to 0.2
parser.add_argument("--top_k", type=int)
parser.add_argument("--top_p", type=float)
parser.add_argument("--max_new_tokens", type=int)
args = parser.parse_args()

cfg = load_config()
for key, value in vars(args).items():                     # vars(args) -> dict of all flags
    if value is not None:                                 # only override flags the user actually passed
        cfg["generation"][key] = value
print(generation_kwargs(cfg))
