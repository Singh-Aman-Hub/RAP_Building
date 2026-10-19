"""Ex 05: a small class + a dataclass config."""
from dataclasses import dataclass


@dataclass
class StackConfig:
    max_size: int = 3


class Stack:
    def __init__(self, config: StackConfig):
        self.config = config
        self.items = []

    def push(self, item):
        if len(self.items) >= self.config.max_size:
            raise OverflowError("stack is full")
        self.items.append(item)

    def pop(self):
        if not self.items:
            raise IndexError("pop from empty stack")
        return self.items.pop()

    def __repr__(self):
        return f"Stack({self.items})"


s = Stack(StackConfig(max_size=2))
s.push(1); s.push(2)
try:
    s.push(3)
except OverflowError as e:
    print("caught:", e)
print(s)
print("popped", s.pop())
print(s)
