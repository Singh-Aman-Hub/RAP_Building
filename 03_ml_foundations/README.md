# 03 — ML Foundations (the bridge to LLMs)

## The story of this module
**Pain point:** a normal program needs rules you write by hand, and nobody can write the rule for "is this review positive?". **Spark:** hold data as arrays (01), let gradient descent find the numbers (02), let autograd compute the gradients (03) and package the whole thing as the five-line loop (04: forward, loss, `zero_grad`, backward, step), the *universal training loop* that later fine-tunes an LLM. **Along the way:** the loop can memorise instead of learn, so we watch validation (05); scores become decisions with softmax and cross-entropy (06) and are judged honestly with metrics (07). **What you build:** a TF-IDF text classifier you train, save, load and test (08). **Cliffhanger:** it works, but `king` and `queen` are unrelated columns (cosine 0.0) and `dog bites man` equals `man bites dog` (`08/flaw_demo.py`). We need a better way to represent text, and first a way to feed raw text in at all: [../04_llm_fundamentals/](../04_llm_fundamentals/) (start with [BRIDGE_FROM_03.md](../04_llm_fundamentals/BRIDGE_FROM_03.md)).

## Why this matters
An LLM is "just" a very large model trained with the loop you write here. If you can explain loss, learning rate, batch size, epochs, overfitting, softmax and a PyTorch training loop on a 3-parameter model, then LoRA, temperature and fine-tuning later are small extensions, not magic. Interviewers also ask for a simple end-to-end pipeline (train → save → load → serve), which step 08 gives you.

## Where this fits
**Builds on:** [../00_python_foundations/](../00_python_foundations/) (functions, classes, `if __name__`, files), [../01_debugging_and_environment/](../01_debugging_and_environment/) (venv, tracebacks, shape errors), [../02_dsa_arrays_hashmaps/](../02_dsa_arrays_hashmaps/) (arrays, dict/Counter thinking, complexity).

**Leads to:**
- [../04_llm_fundamentals/](../04_llm_fundamentals/): softmax (step 06) is how a model turns logits into a next-token distribution; tensors and `nn.Module` (03/04) are how Hugging Face models are built; embeddings are learned parameters like `w`.
- [../05_llm_parameters/](../05_llm_parameters/): learning rate, batch size, epochs, optimizer, loss (steps 02/04); temperature rescales the logits you printed in step 06.
- [../06_llm_training_and_fast_inference/](../06_llm_training_and_fast_inference/): the exact loop from step 04, plus overfitting/early stopping (05) when fine-tuning on small data; parameter counting (`numel`) leads to the GPU-memory arithmetic.
- [../09_api_development/](../09_api_development/) and [../10_ml_system_architecture/](../10_ml_system_architecture/): the saved pipeline from step 08 is the model they serve, version and monitor; metrics (07) are what monitoring tracks.
- [../13_capstone_project/](../13_capstone_project/): reuses `predict_texts`.

## Concept map
```
 data (features X, labels y)                      (01 numpy: shapes, scaling, split)
        |
        v
 model = function with parameters (w, b)          (02 by hand, 04 nn.Module)
        |  forward pass -> predictions
        v
 loss  = how wrong? (MSE regression / cross-entropy classification)    (02, 06)
        |  gradients (by hand 02, autograd 03)
        v
 optimizer: parameter -= learning_rate * gradient                      (02, 04)
        |        controlled by: learning rate, batch size, epochs
        v
 repeat until validation loss is good                                  (04 loop)
        |
        +--> overfitting check: train vs validation loss               (05)
        +--> evaluation metrics: accuracy / precision / recall / F1    (07)
        +--> save artifact (joblib), load, predict -> served later     (08)
```

## Study path
| # | Folder | You learn | Run | Time |
|---|---|---|---|---|
| 01 | [01_numpy_and_data](01_numpy_and_data/) | features/labels, shapes, broadcasting, split | `python main.py` | 60 min |
| 02 | [02_linear_regression_numpy](02_linear_regression_numpy/) | gradient descent, learning rate, epochs, batch size | `python main.py` | 90 min |
| 03 | [03_pytorch_tensors_autograd](03_pytorch_tensors_autograd/) | tensors, autograd, devices | `python main.py` | 60 min |
| 04 | [04_pytorch_training_loop](04_pytorch_training_loop/) | `nn.Module`, `DataLoader`, optimizer, loop | `python main.py` | 90 min |
| 05 | [05_overfitting_train_val](05_overfitting_train_val/) | train vs validation, overfitting | `python main.py` | 45 min |
| 06 | [06_softmax_cross_entropy](06_softmax_cross_entropy/) | logits → probabilities, loss | `python main.py` | 45 min |
| 07 | [07_metrics](07_metrics/) | accuracy/precision/recall/F1 | `python main.py` | 45 min |
| 08 | [08_sklearn_text_classifier](08_sklearn_text_classifier/) | pipeline, joblib save/load, predict, pytest, and the TF-IDF flaw | `python train.py`, `python predict.py "great food"`, then `python flaw_demo.py` | 100 min |

## Explanations
- **What is ML?** Instead of writing rules, you give examples (X, y) and let an algorithm find parameters that map X to y. *Limit:* it generalises only to data like what it saw.
- **Features / labels / parameters:** features are inputs, labels are the right answers, parameters are the numbers learned.
- **Loss:** one number for "how wrong". MSE for numbers, cross-entropy for classes. Training = minimise loss.
- **Gradient descent:** the gradient says which way each parameter increases the loss; step the other way.
- **Learning rate:** step size. Too small → slow; too large → diverge (step 02 shows `lr=1.1` exploding).
- **Epoch / batch size:** one pass / samples per update. Small batch = noisy, cheap memory; large = stable, memory-hungry.
- **Overfitting:** memorising noise. Detect by validation loss rising while train loss falls.
- **Softmax and cross-entropy:** scores → probabilities → penalty for low probability on the truth.
- **Metrics:** accuracy can lie on imbalanced data; precision/recall/F1 say what kind of errors.
- **TF-IDF's flaw:** every word is its own column (king vs queen cosine 0.0) and order is lost; `flaw_demo.py` prints the proof. This motivates tokenizers and embeddings in module 04.
- **Serialization:** `joblib.dump` the *whole* pipeline (vectorizer + model); a model without its preprocessing is useless.

## Common mistakes & gotchas
- Shape mismatches, e.g. `(500,)` vs `(500,1)`; print `.shape` constantly.
- Forgetting `optimizer.zero_grad()`; forgetting `model.eval()` / `torch.no_grad()` at inference.
- Fitting scalers/vectorizers on all data before splitting (leakage).
- Passing probabilities into `F.cross_entropy` (it wants logits).
- Judging a model by training accuracy; trusting `confidence` as correctness.
- Not seeding randomness (`manual_seed`, `default_rng(seed)`) so results can't be compared.

## Self-check
- [ ] I can explain features, labels, parameters, loss
- [ ] I can write gradient descent for y = w·x + b without help
- [ ] I can write the PyTorch loop in the right order and explain each line
- [ ] I can explain learning rate / batch size / epochs with the experiment results
- [ ] I can spot overfitting from two numbers
- [ ] I can compute softmax and cross-entropy by hand
- [ ] I can compute precision/recall/F1 from a confusion matrix
- [ ] I can save/load a pipeline and explain why preprocessing is included

## Interview questions
1. **What is a loss function?** A single number measuring prediction error that training minimises. MSE (regression), cross-entropy (classification). *Trade-off:* the loss you optimise isn't always the metric you care about.
2. **Explain gradient descent.** Compute the derivative of loss with respect to each parameter and move opposite to it by `learning_rate`. *Limit:* can stall in flat regions or diverge if lr is too high.
3. **What does learning rate do?** Scales the step. Show: 0.001 slow, 1.1 diverged.
4. **Batch size trade-offs?** Larger: smoother gradient, fewer updates per epoch, more memory. Smaller: noisy, more updates, regularising effect.
5. **What's an epoch?** One full pass over the training set.
6. **How do you detect overfitting?** Train loss ↓ while validation loss ↑ (or a large gap). Fixes: more data, simpler model, regularisation, dropout, early stopping.
7. **Why a separate validation set?** To estimate performance on unseen data; the model never trains on it. A test set is held out even from tuning.
8. **Write the order of a training step.** forward → loss → `zero_grad` → `backward` → `step`.
9. **`model.train()` vs `model.eval()`?** Switches layers like dropout/batch-norm between training and inference behaviour; use with `torch.no_grad()` to save memory.
10. **What does softmax do?** Converts logits to a probability distribution summing to 1. *Note:* probabilities ≠ calibrated confidence.
11. **Accuracy vs F1?** Accuracy misleads with imbalanced classes; F1 balances precision and recall.
12. **How do you ship a trained model?** Serialise the full pipeline (joblib/torch.save), record version and metrics, load once at server start.
13. **Why `Pipeline`?** Guarantees identical preprocessing in training and serving and prevents leakage during cross-validation.
14. **What is autograd?** PyTorch records operations in a graph and applies the chain rule on `backward()`.
