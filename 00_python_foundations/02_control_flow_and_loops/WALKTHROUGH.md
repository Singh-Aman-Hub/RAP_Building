# Walkthrough 02 — if/else and loops

## Goal
Make decisions (`if`) and repeat work (`for`, `while`) — the skeleton of every training loop and agent loop.

## Run it
```bash
cd 00_python_foundations/02_control_flow_and_loops
python main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| `elif` | else if (other languages), switch/case | No | Python has no classic `switch`; `elif` chains (or `match` in 3.10+) do that job. |
| ternary | conditional expression, inline if | Same | `a if cond else b` is an expression returning a value. |
| `for` loop vs `while` loop | iterate vs repeat-until | No | `for` walks a known sequence; `while` repeats on a condition you must update. |
| `break` vs `continue` | exit vs skip | No | `break` ends the whole loop; `continue` ends only the current iteration. |
| `range` vs `enumerate` vs `zip` | index loop, indexed loop, parallel loop | No | `range` makes numbers, `enumerate` adds a counter to items, `zip` pairs two sequences. |

## Code, step by step

### 1. File header (docstring)  (`main.py` lines 1–11)

```python
"""
GOAL:  Use if/elif/else and for/while loops to make decisions and repeat work.

RUN:   cd 00_python_foundations/02_control_flow_and_loops && python main.py

EXPECTED OUTPUT (first lines):
    loss 2.5 -> still learning
    ...

NEEDS: nothing (standard library only), no internet.
"""
```

- **In simple words:** The file's label: goal, how to run, what to expect.
- **Exactly:** Module docstring; executes nothing.
- **Why it is here:** Orients the reader before any code.

### 2. Define describe_loss with if / elif / else  (`main.py` lines 13–21)

```python
# ---- if / elif / else ----------------------------------------------------
# Indentation (4 spaces) IS the syntax: it marks which lines belong to the block.
def describe_loss(loss):
    if loss > 2.0:                 # condition ends with a colon
        return "still learning"
    elif loss > 0.5:               # elif = "else if"
        return "getting better"
    else:
        return "good"
```

- **In simple words:** A small decision machine: give it a loss number, it answers with a phrase. Indentation shows which lines belong to which branch.
- **Exactly:** `def describe_loss(loss):` defines a function (covered fully in step 03). Python tests top to bottom and runs only the FIRST true branch: `loss > 2.0` -> "still learning"; else `loss > 0.5` -> "getting better"; else "good". Each `if/elif/else` line ends with `:` and the body is indented 4 spaces; `return` exits at once.
- **Why it is here:** Branching is how a program reacts to data. Mixing tabs/spaces or forgetting the colon gives `IndentationError`/`SyntaxError`.

### 3. Call it for three losses  (`main.py` lines 24–25)

```python
for loss in [2.5, 1.0, 0.2]:
    print(f"loss {loss} -> {describe_loss(loss)}")
```

- **In simple words:** Try the decision machine on three example numbers.
- **Exactly:** `for loss in [2.5, 1.0, 0.2]` runs the body 3 times. Real output: `loss 2.5 -> still learning`, `loss 1.0 -> getting better`, `loss 0.2 -> good`. 1.0 is not `> 2.0` but is `> 0.5`, so the middle branch fires.
- **Why it is here:** Shows each branch is reachable and that order of conditions matters.

### 4. Comparison, logic operators and ternary  (`main.py` lines 27–34)

```python
# Comparison operators: ==  !=  <  >  <=  >=   Logic: and  or  not
epochs, lr = 3, 0.001
if epochs > 0 and lr < 0.01:
    print("config looks sane")

# One-line conditional ("ternary"): value_if_true if condition else value_if_false
mode = "train" if epochs > 0 else "eval"
print("mode:", mode)
```

- **In simple words:** `and` needs both sides true. The one-line `x if cond else y` picks one of two values.
- **Exactly:** `epochs, lr = 3, 0.001` is tuple unpacking (two assignments). `3 > 0 and 0.001 < 0.01` is `True` so prints `config looks sane`. `mode = "train" if epochs > 0 else "eval"` -> `'train'`; `print("mode:", mode)` prints `mode: train` (print joins args with a space).
- **Why it is here:** Short guard checks and one-line choices keep config code readable.

### 5. Four ways to loop over a list  (`main.py` lines 36–49)

```python
# ---- for loops -----------------------------------------------------------
print("\n--- for loops ---")
tokens = ["The", "cat", "sat"]
for token in tokens:                      # visits each item in order
    print("token:", token)

for i in range(3):                        # range(3) -> 0, 1, 2  (stop is exclusive)
    print("step", i)

for i, token in enumerate(tokens):        # enumerate gives (index, item) pairs
    print(i, token)

for t, u in zip(["a", "b"], [1, 2]):      # zip walks two lists side by side
    print(t, u)
```

- **In simple words:** Walk through items directly, by counting, with numbers attached, or two lists at once.
- **Exactly:** `for token in tokens` yields `The`, `cat`, `sat`. `range(3)` yields `0,1,2` (stop exclusive). `enumerate(tokens)` yields `(0,'The'),(1,'cat'),(2,'sat')`. `zip(["a","b"],[1,2])` yields `('a',1),('b',2)` and stops at the shorter list.
- **Why it is here:** Prefer these over `for i in range(len(x))`; they are shorter and avoid index mistakes.

### 6. While loop that shrinks the loss  (`main.py` lines 51–58)

```python
# ---- while loops ---------------------------------------------------------
print("\n--- while loop ---")
loss = 3.2
step = 0
while loss > 1.0:                         # repeats until the condition is False
    loss = loss * 0.7                     # pretend training reduces the loss
    step += 1                             # x += 1 is shorthand for x = x + 1
    print(f"step {step}: loss={loss:.3f}")
```

- **In simple words:** Keep repeating while a condition is still true, like stirring until the sugar dissolves.
- **Exactly:** Start `loss=3.2, step=0`. Each pass multiplies by 0.7 and adds 1: 2.240, 1.568, 1.098, 0.768. After step 4, `0.768 > 1.0` is False and the loop ends. `step += 1` means `step = step + 1`. `:.3f` shows 3 decimals.
- **Why it is here:** A `while` needs something inside that changes the condition; otherwise it runs forever.

### 7. continue and break  (`main.py` lines 60–68)

```python
# ---- break / continue ----------------------------------------------------
# This pattern (a loop with a max-steps guard) returns in the agent harness (module 08).
print("\n--- break / continue ---")
for n in range(10):
    if n % 2 == 0:
        continue                          # skip the rest of this iteration
    if n > 6:
        break                             # leave the loop entirely
    print("odd:", n)
```

- **In simple words:** `continue` means skip to the next round; `break` means leave the loop completely.
- **Exactly:** `range(10)` gives 0-9. Even numbers hit `continue` (n % 2 == 0). Odd numbers 1,3,5 print; at 7, `7 > 6` triggers `break`. Real output: `odd: 1`, `odd: 3`, `odd: 5`.
- **Why it is here:** These are the early exits used in search loops and agent loops with a max-step guard.

### 8. for ... else  (`main.py` lines 70–76)

```python
# for ... else : the else block runs only if the loop was NOT ended by break
for n in [2, 4, 6]:
    if n % 2 == 1:
        print("found odd")
        break
else:
    print("no odd numbers found")
```

- **In simple words:** The `else` of a loop runs only if the loop finished without `break`.
- **Exactly:** List `[2,4,6]` has no odd number, so `break` never runs and `else` prints `no odd numbers found`.
- **Why it is here:** A neat way to say "not found" without a flag variable.

### 9. TRY IT and BREAK IT comments  (`main.py` lines 78–80)

```python
# TRY IT: change the 0.7 factor to 0.99 and watch how many steps the while loop needs.
# BREAK IT: set the factor to 1.1 -> infinite loop! Press Ctrl+C: you will see KeyboardInterrupt.
#           (This is why real loops get a max_steps guard.)
```

- **In simple words:** Experiments left for you; comments only.
- **Exactly:** Changing 0.7 to 1.1 makes `loss` grow, `loss > 1.0` never becomes False: infinite loop, stop with Ctrl+C (`KeyboardInterrupt`).
- **Why it is here:** Shows why real loops carry a `max_steps` guard.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `if cond:` / `elif` / `else:` | Run a block only when the condition is True; indentation = block | `if loss > 2: ...` |
| `and`, `or`, `not` | Boolean logic | `epochs > 0 and lr < 0.01` |
| `a if cond else b` | One-line conditional expression | `"train" if epochs > 0 else "eval"` |
| `for x in items:` | Visit each item | `for t in tokens:` |
| `range(n)` | Numbers `0 .. n-1` (stop excluded) | `range(3)` -> 0,1,2 |
| `enumerate(items)` | Gives `(index, item)` | `for i, t in enumerate(tokens)` |
| `zip(a, b)` | Walks two sequences together | `zip(["a"], [1])` |
| `while cond:` | Repeat while True | loss-decay loop |
| `x += 1` | Shorthand for `x = x + 1` | `step += 1` |
| `break` / `continue` | Leave loop / skip to next iteration | see main.py |
| `for ... else` | `else` runs if no `break` happened | "no odd numbers found" |

## Libraries & functions
All built-ins: `range`, `enumerate`, `zip`, `print`. No third-party library.
| Name | What it does | Returns |
|---|---|---|
| `range(start, stop, step)` | Lazy sequence of ints | `range` object (iterable) |
| `enumerate(iterable, start=0)` | Pairs index with item | iterator of tuples |
| `zip(a, b)` | Pairs items; stops at the shorter | iterator of tuples |

## Execution trace (real output)
```
loss 2.5 -> still learning      (2.5 > 2.0)
loss 1.0 -> getting better      (not > 2.0, but > 0.5)
loss 0.2 -> good                (else branch)
```
While loop (loss starts 3.2, multiplied by 0.7 each pass until it is <= 1.0):
```
step 1: loss=2.240
step 2: loss=1.568
step 3: loss=1.098
step 4: loss=0.768      <- 0.768 is not > 1.0, loop ends
```
`break/continue` demo prints `odd: 1`, `odd: 3`, `odd: 5` (evens skipped; 7 > 6 triggers `break`).

## Experiments
1. Change `0.7` to `0.99`: many more steps (about 116) — slow learning rate analogue.
2. Change `n > 6` to `n > 2` in the break demo: prints only `odd: 1`.
3. Replace `range(3)` with `range(1, 10, 3)`: prints 1, 4, 7.

## Debugging practice
Set the factor to `1.1` and run: the loss grows so `loss > 1.0` never becomes False — an infinite loop.
Press `Ctrl+C`; the traceback ends in `KeyboardInterrupt` and shows which line you were stuck on. Lesson: every loop that depends on external progress (training, agents, retries) needs a max-step guard — you will write one in [module 08](../../08_agent_harness/).

## Rebuild checklist
- [ ] Function returning "low/medium/high" with if/elif/else
- [ ] `for` with `enumerate` printing `index: item`
- [ ] `while` loop with a `max_steps` safety counter

## Say it aloud
"`for` loops iterate over a known collection, `while` loops repeat until a condition changes. I use `enumerate` when I need the index and `zip` for parallel lists. `break` exits early and `continue` skips one pass. A `while` loop can run forever if the condition never changes, so I add a step counter."
Trade-off: `while` is flexible but easy to make infinite; prefer `for` when the number of items is known.
