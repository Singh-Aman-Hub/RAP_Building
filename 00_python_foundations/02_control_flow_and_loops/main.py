"""
GOAL:  Use if/elif/else and for/while loops to make decisions and repeat work.

RUN:   cd 00_python_foundations/02_control_flow_and_loops && python main.py

EXPECTED OUTPUT (first lines):
    loss 2.5 -> still learning
    ...

NEEDS: nothing (standard library only), no internet.
"""

# ---- if / elif / else ----------------------------------------------------
# Indentation (4 spaces) IS the syntax: it marks which lines belong to the block.
def describe_loss(loss):
    if loss > 2.0:                 # condition ends with a colon
        return "still learning"
    elif loss > 0.5:               # elif = "else if"
        return "getting better"
    else:
        return "good"


for loss in [2.5, 1.0, 0.2]:
    print(f"loss {loss} -> {describe_loss(loss)}")

# Comparison operators: ==  !=  <  >  <=  >=   Logic: and  or  not
epochs, lr = 3, 0.001
if epochs > 0 and lr < 0.01:
    print("config looks sane")

# One-line conditional ("ternary"): value_if_true if condition else value_if_false
mode = "train" if epochs > 0 else "eval"
print("mode:", mode)

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

# ---- while loops ---------------------------------------------------------
print("\n--- while loop ---")
loss = 3.2
step = 0
while loss > 1.0:                         # repeats until the condition is False
    loss = loss * 0.7                     # pretend training reduces the loss
    step += 1                             # x += 1 is shorthand for x = x + 1
    print(f"step {step}: loss={loss:.3f}")

# ---- break / continue ----------------------------------------------------
# This pattern (a loop with a max-steps guard) returns in the agent harness (module 08).
print("\n--- break / continue ---")
for n in range(10):
    if n % 2 == 0:
        continue                          # skip the rest of this iteration
    if n > 6:
        break                             # leave the loop entirely
    print("odd:", n)

# for ... else : the else block runs only if the loop was NOT ended by break
for n in [2, 4, 6]:
    if n % 2 == 1:
        print("found odd")
        break
else:
    print("no odd numbers found")

# TRY IT: change the 0.7 factor to 0.99 and watch how many steps the while loop needs.
# BREAK IT: set the factor to 1.1 -> infinite loop! Press Ctrl+C: you will see KeyboardInterrupt.
#           (This is why real loops get a max_steps guard.)
