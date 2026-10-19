"""
common.py -- small helpers shared by the steps of module 06 (so each step stays short).

You build these same ideas by hand in 01_dataset_preparation; here they are packaged.
"""

from pathlib import Path

import torch

MODULE_DIR = Path(__file__).resolve().parent            # .../06_llm_training_and_fast_inference
CORPUS_PATH = MODULE_DIR / "data" / "tiny_corpus.txt"   # 40 customer-support lines for a made-up app


def get_device() -> torch.device:
    """Pick the fastest device we have: Apple GPU (mps) > NVIDIA GPU (cuda) > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_lines(path: Path = CORPUS_PATH) -> list[str]:
    """Return the non-empty lines of the corpus file."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_blocks(tokenizer, lines: list[str], block_size: int = 48) -> torch.Tensor:
    """Tokenize all lines, glue them together (separated by EOS) and cut into equal blocks.

    Returns a tensor of shape (num_blocks, block_size) holding token IDs.
    """
    ids: list[int] = []
    for line in lines:
        ids.extend(tokenizer.encode(line) + [tokenizer.eos_token_id])  # EOS marks "document ends here"
    n_blocks = len(ids) // block_size                    # throw away the incomplete tail
    return torch.tensor(ids[: n_blocks * block_size]).view(n_blocks, block_size)
