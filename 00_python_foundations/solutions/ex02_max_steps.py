"""Ex 02: a while loop with a max_steps guard (the pattern reused by the agent harness)."""
loss, step, max_steps = 3.2, 0, 50
while loss > 1.0 and step < max_steps:
    loss *= 1.1            # BAD "training": loss grows, so only the guard stops us
    step += 1
print("stopped at step", step, "loss", round(loss, 2),
      "-> hit the guard" if step == max_steps else "-> converged")
