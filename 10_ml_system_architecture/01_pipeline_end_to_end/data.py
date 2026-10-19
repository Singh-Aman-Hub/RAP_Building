"""
data.py -- a tiny, deterministic sentiment dataset used by every step in module 10.

(Module 03 builds a text classifier too; this file keeps module 10 self-contained and runnable
on its own. The ideas -- collect, clean, vectorize, train, evaluate -- are identical.)
"""

import random

NOUNS = ["movie", "phone", "service", "food", "book", "hotel", "app", "game", "teacher", "delivery"]
POS = ["great", "excellent", "amazing", "wonderful", "fantastic", "lovely", "brilliant", "superb"]
NEG = ["terrible", "awful", "horrible", "disappointing", "dreadful", "poor", "boring", "worthless"]
TEMPLATES = ["The {n} was {a}.", "I thought the {n} was really {a}!", "What a {a} {n}.", "Honestly, a {a} {n}."]

# Handwritten sentences the model never trains on -> an honest test set.
TEST_SET = [
    ("The hotel was lovely and the staff were amazing", "positive"),
    ("Absolutely brilliant game, loved every minute", "positive"),
    ("Superb food and excellent service", "positive"),
    ("Fantastic delivery, arrived early", "positive"),
    ("The book was boring and awful", "negative"),
    ("Terrible app, it keeps crashing, worthless", "negative"),
    ("Horrible service and dreadful food", "negative"),
    ("A poor, disappointing phone", "negative"),
]


def get_training_data(n: int = 200, seed: int = 0) -> tuple[list[str], list[str]]:
    """Generate n labelled sentences from templates. Same seed -> same data (reproducibility!)."""
    rng = random.Random(seed)                      # a private random generator so we never disturb global state
    texts, labels = [], []
    for _ in range(n):
        positive = rng.random() < 0.5              # coin flip decides the class
        adjective = rng.choice(POS if positive else NEG)
        text = rng.choice(TEMPLATES).format(n=rng.choice(NOUNS), a=adjective)
        texts.append(text)
        labels.append("positive" if positive else "negative")
    return texts, labels


def get_test_data() -> tuple[list[str], list[str]]:
    texts, labels = zip(*TEST_SET)                 # unzip list of pairs into two tuples
    return list(texts), list(labels)
