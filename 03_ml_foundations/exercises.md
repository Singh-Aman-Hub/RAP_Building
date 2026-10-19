# Exercises — module 03

Change ONE thing at a time. Worked answers: [solutions/](solutions/).

| Step | Modify it yourself | Break it & read the error | Rebuild from memory | Explain aloud |
|---|---|---|---|---|
| 01 | Standardise `y` too; compute MSE on the scaled scale. Make a 10-sample dataset and split 70/30. | Mismatched matmul shapes | MSE for a line | "What is a feature vs a label?" |
| 02 | Add momentum: `v = 0.9*v - lr*grad; w += v`. Plot loss (optional matplotlib). Find the largest stable lr. | Flip the update sign | The 4-line update (forward/loss/grad/update) | "What does an epoch do?" |
| 03 | Use autograd to fit `y = 3x - 2`. | Remove `no_grad` | `loss.backward()` + read `.grad` | "Why zero gradients?" |
| 04 | Swap in `Adam`; add a hidden layer; compare val loss at epoch 3 (solutions/ex04_adam_hidden_layer.py). | Remove `zero_grad`; pass wrong `y` shape | The 5-line loop body | "train vs eval mode?" |
| 05 | Add L2 regularisation via `np.polyfit` on scaled features, or reduce degree; find the best degree by validation error. | Evaluate val on train | Overfitting table logic | "How do I detect overfitting?" |
| 06 | Add a 4th class and recompute probabilities/loss. Implement `log_softmax` stably. | Double softmax with `F.cross_entropy` | softmax + CE | "Why does -log(p) punish confident mistakes?" |
| 07 | Compute macro-F1; build a 3-class confusion matrix by hand (solutions/ex07_macro_f1.py). | Precision when no positives | P/R/F1 formulas | "Which metric for cancer screening?" |
| 08 | In `flaw_demo.py` add `"the monarch rules the land"`; predict the king/monarch cosine before running. Add a third label "neutral" with 20 sentences; retrain; change version to v2 and compare metadata. Add a `/batch` style function that takes many texts. | Run predict before train; look up a missing word in `vocab` (`KeyError`) | Pipeline + joblib save/load | "Why save the vectorizer with the model? Why do king and queen have cosine 0.0?" |

## Cumulative challenge (45 min, no AI)
1. Generate `y = 4x − 2 + noise`, train with PyTorch (`nn.Linear(1,1)`), report train/val loss.
2. Try lr in {0.001, 0.1, 5}; write one sentence on each outcome.
3. Save with `torch.save(model.state_dict(), "m.pt")`, load into a new model, confirm identical predictions.
