"""Solution ex02: token report for any text (words, tokens, ratio, rough cost).
RUN: .venv/bin/python 04_llm_fundamentals/solutions/ex02_token_report.py "your text here"  (argument optional)"""
import sys
from transformers import AutoTokenizer


def token_report(text, tokenizer, price_per_million_tokens=0.5):
    ids = tokenizer.encode(text)
    words = len(text.split())
    return {
        "characters": len(text),
        "words": words,
        "tokens": len(ids),
        "tokens_per_word": round(len(ids) / max(words, 1), 2),
        "approx_cost_usd": round(len(ids) / 1e6 * price_per_million_tokens, 8),
        "pieces": tokenizer.convert_ids_to_tokens(ids),
    }


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    # sys.argv[0] is the script name; anything after it is a command-line argument
    text = " ".join(sys.argv[1:]) or "Tokenization of uncommonly-spelled words costs more than plain English."
    for key, value in token_report(text, tok).items():
        print(f"{key:16}: {value}")
