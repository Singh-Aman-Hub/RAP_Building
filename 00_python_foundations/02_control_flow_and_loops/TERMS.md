# Terms — Control Flow and Loops

Folder: `00_python_foundations/02_control_flow_and_loops` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [00 Python foundations README](../README.md)

**Terms in this folder:** [Indentation](#indentation) · [def (function definition)](#def-function-definition) · [if / elif / else](#if--elif--else) · [Comparison operators](#comparison-operators) · [Logical operators](#logical-operators-and-or-not) · [Conditional expression](#conditional-expression-ternary) · [for loop](#for-loop) · [range()](#range) · [enumerate()](#enumerate) · [zip()](#zip) · [while loop](#while-loop) · [Augmented assignment](#augmented-assignment-) · [continue](#continue) · [break](#break) · [for ... else](#for--else) · [Modulo operator](#modulo-operator-) · [Infinite loop](#infinite-loop) · [KeyboardInterrupt](#keyboardinterrupt) · [Max-steps guard](#max-steps-guard)

---

## Indentation  `(syntax)`
**1. Definition:** In Python, the leading whitespace (conventionally 4 spaces) of a line defines which block of code it belongs to.
**2. In plain words:** Other languages use braces `{}`; Python uses spacing. Lines pushed in under `if loss > 2.0:` run only when that condition is true. Mixing tabs and spaces or mis-aligning a line causes an `IndentationError`.
**3. Example to quote:**
> "In `describe_loss`, the `return` lines are indented under each `if`, so only the branch that matches runs."
```python
if True:
    print("inside")
print("outside")
```
**Seen in:** `main.py`, `describe_loss` and every loop

## def (function definition)  `(syntax)`
**1. Definition:** `def name(params):` defines a reusable function whose body is the indented block below it; `return` sends a value back to the caller.
**2. In plain words:** A function packages a few lines under a name, so you can call it with different inputs. Here `describe_loss(loss)` turns a number into a phrase. This step only uses it as a container for `if/elif/else`; functions are covered fully in step 03.
**3. Example to quote:**
> "I wrapped the loss checks in `describe_loss(loss)` so the loop can call it for 2.5, 1.0 and 0.2."
```python
def describe_loss(loss):
    return "good" if loss < 0.5 else "keep training"
```
**Seen in:** `main.py`, top of file

## if / elif / else  `(syntax)`
**1. Definition:** `if` runs its block when the condition is true; `elif` ("else if") is tested only if the earlier conditions were false; `else` runs when none matched.
**2. In plain words:** Python checks the conditions from top to bottom and runs only the first matching block. For `loss = 1.0`, `loss > 2.0` is false, so it moves to `loss > 0.5`, which is true, and returns "getting better". Order matters: put the most specific test first.
**3. Example to quote:**
> "`describe_loss(2.5)` says 'still learning' because the first condition `loss > 2.0` matched, and the other branches were skipped."
```python
x = 1.0
if x > 2.0: print("a")
elif x > 0.5: print("b")
else: print("c")   # prints b
```
**Seen in:** `main.py`, `describe_loss`

## Comparison operators  `(syntax)`
**1. Definition:** `==`, `!=`, `<`, `>`, `<=`, `>=` compare two values and produce a `bool`.
**2. In plain words:** They answer questions like "is the loss bigger than 2?". The classic mistake is writing `=` (assignment) when you mean `==` (comparison). Comparing floats with `==` is risky because of rounding.
**3. Example to quote:**
> "`lr < 0.01` checks the learning rate is small; it evaluates to `True` for 0.001."
```python
print(0.001 < 0.01)   # True
```
**Seen in:** `main.py`, `if epochs > 0 and lr < 0.01`

## Logical operators (and, or, not)  `(syntax)`
**1. Definition:** `and` is true if both sides are true, `or` if at least one is, and `not` inverts a boolean.
**2. In plain words:** They combine conditions. `epochs > 0 and lr < 0.01` means "both must hold". Python stops early: if the left side of `and` is false it never checks the right side.
**3. Example to quote:**
> "My config check uses `epochs > 0 and lr < 0.01`, and it printed 'config looks sane' because both parts were true."
```python
print(True and not False)   # True
```
**Seen in:** `main.py`, config sanity check

## Conditional expression (ternary)  `(syntax)`
**1. Definition:** `a if condition else b` is an expression that evaluates to `a` when the condition is true and to `b` otherwise.
**2. In plain words:** It's a one-line `if/else` for choosing between two values. `mode = "train" if epochs > 0 else "eval"` sets `mode` in a single line. Use it for simple choices only; long ones are better as normal `if` blocks.
**3. Example to quote:**
> "I set `mode` with one line: `\"train\" if epochs > 0 else \"eval\"` and it printed `mode: train`."
```python
mode = "train" if 3 > 0 else "eval"
```
**Seen in:** `main.py`, `mode`

## for loop  `(syntax)`
**1. Definition:** `for item in iterable:` executes its block once for each element produced by the iterable, in order.
**2. In plain words:** Use `for` when you have a collection or a known count, like a list of tokens. The loop variable (`token`) holds the current item. It's the safest loop because it ends by itself.
**3. Example to quote:**
> "`for token in tokens:` visited 'The', 'cat', 'sat' one at a time and printed each."
```python
for token in ["The", "cat", "sat"]:
    print(token)
```
**Seen in:** `main.py`, for loops section

## range()  `(Python built-in)`
**1. Definition:** `range(start, stop, step)` creates a lazy immutable sequence of integers from `start` up to but excluding `stop`; with one argument it starts at 0.
**2. In plain words:** `range(3)` gives 0, 1, 2. It doesn't build a big list in memory; it produces numbers when asked. The stop value is never included, which is why `range(3)` has three items and not four.
**3. Example to quote:**
> "`for i in range(3)` printed `step 0`, `step 1`, `step 2`, which matches how my training steps are counted from zero."
```python
print(list(range(1, 10, 3)))   # [1, 4, 7]
```
**Seen in:** `main.py`, `for i in range(3)`; WALKTHROUGH experiment 3

## enumerate()  `(Python built-in)`
**1. Definition:** `enumerate(iterable, start=0)` returns an iterator of `(index, item)` pairs.
**2. In plain words:** When you need both the position and the value, `enumerate` is cleaner than manually keeping a counter. It's common for numbering tokens or printing line numbers.
**3. Example to quote:**
> "`for i, token in enumerate(tokens)` printed `0 The`, `1 cat`, `2 sat`, so I know each token's position."
```python
for i, t in enumerate(["a", "b"], start=1):
    print(i, t)   # 1 a / 2 b
```
**Seen in:** `main.py`, for loops section

## zip()  `(Python built-in)`
**1. Definition:** `zip(a, b, ...)` returns an iterator of tuples pairing the items at the same position of each input, stopping at the shortest.
**2. In plain words:** It walks two lists side by side like the teeth of a zipper. Pairing `["a", "b"]` with `[1, 2]` yields `("a", 1)` and `("b", 2)`. If one list is longer, the extra items are silently dropped.
**3. Example to quote:**
> "I used `zip(tokens, ids)` to print each token next to its id without index juggling."
```python
for t, u in zip(["a", "b"], [1, 2]):
    print(t, u)
```
**Seen in:** `main.py`, `zip(["a", "b"], [1, 2])`

## while loop  `(syntax)`
**1. Definition:** `while condition:` repeats its block as long as the condition stays true, re-checking before each pass.
**2. In plain words:** Use `while` when you don't know in advance how many repeats you need, like "train until the loss drops below 1.0". The danger is that if nothing in the loop changes the condition, it never stops.
**3. Example to quote:**
> "My loop multiplied `loss` by 0.7 each pass and stopped at step 4 when `loss` hit 0.768, because `0.768 > 1.0` was false."
```python
loss = 3.2
while loss > 1.0:
    loss *= 0.7
```
**Seen in:** `main.py`, while loops section

## Augmented assignment (+=)  `(syntax)`
**1. Definition:** `x += 1` is shorthand for `x = x + 1`; similar forms exist for `-=`, `*=`, `/=`.
**2. In plain words:** It updates a variable using its own current value. `step += 1` is the standard way to count loop iterations. `loss = loss * 0.7` can also be written `loss *= 0.7`.
**3. Example to quote:**
> "`step += 1` increments my counter each pass, so I can print 'step 1', 'step 2' and so on."
```python
step = 0
step += 1   # step is now 1
```
**Seen in:** `main.py`, while loop

## continue  `(syntax)`
**1. Definition:** `continue` skips the rest of the current loop iteration and jumps to the next one.
**2. In plain words:** It says "ignore this item and move on". In the demo, even numbers hit `continue`, so only odd numbers reach the `print`.
**3. Example to quote:**
> "Even values of `n` triggered `continue`, so my loop only printed `odd: 1`, `odd: 3`, `odd: 5`."
```python
for n in range(4):
    if n % 2 == 0:
        continue
    print(n)   # 1, 3
```
**Seen in:** `main.py`, break / continue section

## break  `(syntax)`
**1. Definition:** `break` immediately exits the innermost enclosing loop.
**2. In plain words:** Use it to stop early once you've found what you want or hit a limit. In the demo, `n = 7` is greater than 6, so `break` ends the loop and 7, 8, 9 are never visited.
**3. Example to quote:**
> "I added `if n > 6: break` so the loop stops as soon as the limit is passed, the same way an agent stops at a max step."
```python
for n in range(10):
    if n > 2:
        break
```
**Seen in:** `main.py`, break / continue section

## for ... else  `(syntax)`
**1. Definition:** A loop's `else` block runs only if the loop finished normally, i.e. without hitting `break`.
**2. In plain words:** It reads as "if the search did not find anything". With `[2, 4, 6]` no odd number triggers `break`, so the `else` prints "no odd numbers found". It's rarely used but a nice interview detail.
**3. Example to quote:**
> "My `for n in [2, 4, 6]` found no odd number, so `break` never ran and the `else` branch printed 'no odd numbers found'."
```python
for n in [2, 4]:
    if n % 2: break
else:
    print("none")
```
**Seen in:** `main.py`, end of file

## Modulo operator (%)  `(syntax)`
**1. Definition:** `a % b` returns the remainder of dividing `a` by `b`.
**2. In plain words:** `n % 2 == 0` is the standard "is this number even?" test. It is also used for "do something every N steps", such as logging every 100 steps.
**3. Example to quote:**
> "I check `n % 2 == 0` to skip even numbers; 7 % 2 is 1, so 7 counts as odd."
```python
print(7 % 2)   # 1
```
**Seen in:** `main.py`, `if n % 2 == 0`

## Infinite loop  `(concept)`
**1. Definition:** An infinite loop is a loop whose condition never becomes false (or that has no exit), so it never ends on its own.
**2. In plain words:** If the factor in the while loop is 1.1, the loss grows and `loss > 1.0` is always true, so the program hangs. It looks like the computer froze. Stop it with Ctrl+C and add a guard so it can't happen again.
**3. Example to quote:**
> "When I changed 0.7 to 1.1 the loss kept growing, the `while` condition never turned false, and I had to press Ctrl+C."
```python
# while True:   # runs forever unless you break
#     pass
```
**Seen in:** `main.py` BREAK IT comment; `WALKTHROUGH.md` debugging practice

## KeyboardInterrupt  `(Python built-in)`
**1. Definition:** `KeyboardInterrupt` is the exception raised when the user presses Ctrl+C while a program is running.
**2. In plain words:** It's how you stop a stuck script. The traceback it prints shows the exact line the program was on, which tells you where the loop was spinning. It is a useful debugging clue.
**3. Example to quote:**
> "After Ctrl+C the traceback ended in `KeyboardInterrupt` and pointed at the `loss = loss * 1.1` line, showing me which loop was stuck."
```python
# Press Ctrl+C during a long loop to raise KeyboardInterrupt
```
**Seen in:** `main.py` BREAK IT comment

## Max-steps guard  `(concept)`
**1. Definition:** A max-steps guard is a counter with an upper limit that forces a loop to stop even if its normal exit condition is never met.
**2. In plain words:** For anything that depends on outside progress (training, agent loops, retries), add `step < max_steps` so a bug can't spin forever. You will reuse this exact pattern in the agent harness.
**3. Example to quote:**
> "I wrote `while loss > 1.0 and step < 100:` so even a bad factor stops after 100 steps."
```python
step = 0
while step < 100:
    step += 1
```
**Seen in:** `main.py` comment about module 08; `WALKTHROUGH.md` rebuild checklist
