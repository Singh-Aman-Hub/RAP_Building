# Step 06 — Softmax and cross-entropy

## Goal
Turn scores into probabilities and score them. Same maths as next-token prediction in an LLM.

## Run it
```bash
python main.py
.venv/bin/python 03_ml_foundations/06_softmax_cross_entropy/main.py
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| default arg `temperature=1.0` | optional parameter | `softmax(z, 0.5)` |
| `z = z - z.max()` | reassign to a shifted copy | stability trick |
| `probs[true_class]` | index with an int | |
| `.unsqueeze(0)` | add batch dimension: (3,) → (1,3) | cross_entropy wants (batch, classes) |

## Libraries & functions
| Name | Origin | Does | Notes |
|---|---|---|---|
| `np.exp`, `np.log` | numpy | exponential / natural log | |
| `F.cross_entropy(logits, target)` | torch.nn.functional | softmax + log + negative mean | takes RAW logits, class indices |

## Execution trace (real run)
Logits `[2.0, 1.0, 0.1]` → exp `[7.39, 2.72, 1.11]` → sum 11.21 → probs `[0.659 0.242 0.099]`.
```
loss if truth=cat : 0.417          (-ln 0.659)
loss if truth=bird: 2.317          (-ln 0.099)
torch cross_entropy: 0.417 (matches numpy)
temperature 0.5: [0.864 0.117 0.019]
temperature 1.0: [0.659 0.242 0.099]
temperature 5.0: [0.4   0.327 0.273]
```
Low temperature sharpens; high flattens. Module [05](../../05_llm_parameters/) uses exactly this for text generation.

## Experiments
1. Logits `[10, 1, 0.1]`: loss near 0.
2. Make all logits equal → uniform probabilities, loss = ln(3) = 1.0986.
3. Temperature 0.1 and 100.

## Debugging practice
Passing probabilities to `F.cross_entropy` produces a wrong number with no error (double softmax). Rule: loss functions with "cross_entropy" in PyTorch want logits.

## Rebuild checklist
- [ ] Softmax with the max-subtraction trick.
- [ ] Cross-entropy for one example.
- [ ] Explain why confident-and-wrong is punished heavily.

## Say it aloud
"Softmax converts raw scores to probabilities that sum to 1. Cross-entropy is the negative log of the probability given to the correct answer, so it is small when the model is right and confident, and huge when confidently wrong. A language model applies this over its whole vocabulary for every next token."
