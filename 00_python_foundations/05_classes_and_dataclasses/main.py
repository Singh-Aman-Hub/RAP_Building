"""
GOAL:  Understand classes (objects with data + behaviour) because every ML library is built from them.

RUN:   cd 00_python_foundations/05_classes_and_dataclasses && python main.py

EXPECTED OUTPUT (first lines):
    Counter 'clicks' = 3
    ...

NEEDS: nothing (standard library only), no internet.
"""

from dataclasses import dataclass, field  # dataclass: auto-writes __init__/__repr__ for data holders


# ---- a plain class -------------------------------------------------------
class Counter:
    """class = blueprint.  An *instance* (object) is one thing built from it."""

    def __init__(self, name):          # __init__ runs when you call Counter("clicks")
        self.name = name               # `self` = this particular object; self.x stores data on it
        self.value = 0

    def increment(self, by=1):         # a method = function defined inside a class
        self.value += by
        return self                    # returning self allows chaining (shown below)

    def __repr__(self):                # controls how print() shows the object
        return f"Counter {self.name!r} = {self.value}"


c = Counter("clicks")                  # create an instance (calls __init__)
c.increment().increment().increment()  # method chaining
print(c)
print(c.value, c.name)                 # attributes are read with dot notation


# ---- inheritance: a class that extends another --------------------------
class LoudCounter(Counter):
    def increment(self, by=1):
        print("incrementing!")
        return super().increment(by)   # super() calls the parent's version


LoudCounter("loud").increment(5)

# ---- how ML libraries look from the outside ------------------------------
# model = AutoModelForCausalLM.from_pretrained("gpt2")   <- a CLASSMETHOD that returns an instance
# out = model.generate(...)                              <- a method on that instance
# model.config.vocab_size                                <- an attribute holding another object
# Below is a tiny imitation so the syntax stops looking magical.
class TinyModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    @classmethod                       # decorator: method receives the class (cls), not an instance
    def from_pretrained(cls, name):
        print(f"(pretend we downloaded {name})")
        return cls(vocab_size=50257)   # cls(...) is the same as TinyModel(...)

    def __call__(self, text):          # makes the instance callable: model("hi")
        return f"output for {text!r}"


model = TinyModel.from_pretrained("gpt2")
print(model.vocab_size, model("hi"))

# ---- dataclass: classes that mostly just hold values ---------------------
@dataclass
class TrainConfig:
    learning_rate: float = 1e-4        # field: type hint + default value
    epochs: int = 3
    batch_size: int = 8
    tags: list = field(default_factory=list)   # mutable defaults need default_factory


cfg = TrainConfig(epochs=5)            # override one value, keep others
print(cfg)                             # free, readable __repr__
print(cfg.learning_rate, cfg == TrainConfig(epochs=5))  # free __eq__

# TRY IT: add a `weight_decay: float = 0.01` field and print it.
# BREAK IT: remove `self.` in `self.value += by` -> UnboundLocalError. Read which line it names.
