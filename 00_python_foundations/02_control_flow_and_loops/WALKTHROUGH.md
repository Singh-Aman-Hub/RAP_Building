# Walkthrough 02 — if/else and loops

## Goal
Make decisions (`if`) and repeat work (`for`, `while`) — the skeleton of every training loop and agent loop.

## Run it
```bash
cd 00_python_foundations/02_control_flow_and_loops
python main.py
```

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
