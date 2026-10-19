# Terms — Classification metrics

Folder: `03_ml_foundations/07_metrics` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [ML foundations README](../README.md)

**Terms in this folder:** [Classification](#classification--concept) · [Positive class](#positive-class--concept) · [True positive, false positive, false negative, true negative](#true-positive-false-positive-false-negative-true-negative--concept) · [Confusion matrix](#confusion-matrix--third-party-scikit-learn) · [Accuracy](#accuracy--concept) · [Precision](#precision--concept) · [Recall](#recall--concept) · [F1 score](#f1-score--concept) · [Imbalanced data](#imbalanced-data--concept) · [scikit-learn metrics functions](#scikit-learn-metrics-functions--third-party-scikit-learn) · [zip()](#zip--python-built-in) · [Counting with sum() and a condition](#counting-with-sum-and-a-condition--syntax) · [List repetition and concatenation](#list-repetition-and-concatenation--syntax) · [ZeroDivisionError / zero_division](#zerodivisionerror--zero_division--concept) · [f-string formatting (`:.2f`)](#f-string-formatting-2f--syntax)

---

## Classification  `(concept)`
**1. Definition:** Classification is a machine-learning task where the model assigns each input to one of a fixed set of categories (classes).
**2. In plain words:** Spam or not spam, positive or negative review, cat or dog. The metrics in this folder measure how well such a model does. (Predicting a number instead is regression.)
**3. Example to quote:**
> "My spam filter is a binary classifier: it labels each email 1 for spam or 0 for not spam."

**Seen in:** `main.py`, `y_true`, `y_pred`

## Positive class  `(concept)`
**1. Definition:** The positive class is the class of interest that the metrics focus on, conventionally labelled 1.
**2. In plain words:** Here "positive" means spam (or fraud, or disease), not "good". Precision and recall are defined relative to it.
**3. Example to quote:**
> "In the spam example, 1 is the positive class: the thing I am trying to catch."

**Seen in:** `main.py`, comment on `y_true`

## True positive, false positive, false negative, true negative  `(concept)`
**1. Definition:** TP: predicted positive and actually positive; FP: predicted positive but actually negative; FN: predicted negative but actually positive; TN: predicted negative and actually negative.
**2. In plain words:** The four possible outcomes of a prediction. A false positive is a false alarm (a real email in the spam folder); a false negative is a miss (spam in the inbox). With the sample data: TP=2, FP=1, FN=2, TN=5.
**3. Example to quote:**
> "TP=2 means I caught 2 spam emails; FN=2 means I missed 2; FP=1 means I wrongly flagged one legitimate email."

**Seen in:** `main.py`, `tp`, `fp`, `fn`, `tn`

## Confusion matrix  `(third-party: scikit-learn)`
**1. Definition:** A confusion matrix is a table of counts of true vs predicted classes; scikit-learn's layout for binary is `[[TN, FP], [FN, TP]]`.
**2. In plain words:** One glance shows what kinds of mistakes the model makes, not only how many. Rows are the truth and columns are predictions.
**3. Example to quote:**
> "`confusion_matrix` printed `[[5, 1], [2, 2]]`: 5 true negatives, 1 false positive, 2 false negatives, 2 true positives."

**Seen in:** `main.py`, `confusion_matrix(y_true, y_pred)`

## Accuracy  `(concept)`
**1. Definition:** Accuracy is the fraction of all predictions that are correct: `(TP + TN) / total`.
**2. In plain words:** The simplest score, but misleading when classes are imbalanced. Here it is 0.70 (7 of 10 right).
**3. Example to quote:**
> "My accuracy is 0.70, but on imbalanced data a model that always says 'no' can reach 99% accuracy, so accuracy alone is not enough."

**Seen in:** `main.py`, `accuracy`

## Precision  `(concept)`
**1. Definition:** Precision is the fraction of predicted positives that are truly positive: `TP / (TP + FP)`.
**2. In plain words:** "Of everything I flagged, how much was right?" High precision means few false alarms. Prioritise it when false alarms are costly, such as a spam filter that must not lose real mail. Here it is 2/3 = 0.67.
**3. Example to quote:**
> "I flagged 3 emails as spam and 2 were really spam, so precision is 2/3, about 0.67."

**Seen in:** `main.py`, `precision`

## Recall  `(concept)`
**1. Definition:** Recall (sensitivity) is the fraction of actual positives that the model found: `TP / (TP + FN)`.
**2. In plain words:** "Of all real spam, how much did I catch?" High recall means few misses. Prioritise it when misses are costly, like cancer screening. Here it is 2/4 = 0.50.
**3. Example to quote:**
> "There were 4 real spam emails and I caught 2, so recall is 0.50."

**Seen in:** `main.py`, `recall`

## F1 score  `(concept)`
**1. Definition:** F1 is the harmonic mean of precision and recall: `2 * P * R / (P + R)`.
**2. In plain words:** It is high only when both precision and recall are high; a model with great precision and terrible recall gets a low F1. Here: 0.57.
**3. Example to quote:**
> "With precision 0.67 and recall 0.50, F1 is `2*0.67*0.50/(0.67+0.50)` = 0.57."

**Seen in:** `main.py`, `f1`

## Imbalanced data  `(concept)`
**1. Definition:** Imbalanced data has far more samples of one class than another, such as 1 fraud in 100 transactions.
**2. In plain words:** A lazy model that always predicts the majority class gets high accuracy but zero usefulness. Use precision, recall and F1 on the minority class instead.
**3. Example to quote:**
> "A model that always says 'not fraud' is 99% accurate but has 0.0 recall, so it catches no fraud at all."

**Seen in:** `main.py`, "always-no" demo

## scikit-learn metrics functions  `(third-party: scikit-learn)`
**1. Definition:** `accuracy_score`, `precision_score`, `recall_score` and `f1_score` from `sklearn.metrics` compute the standard metrics from `y_true` and `y_pred`.
**2. In plain words:** In real projects you use these after learning how they work by hand, and the numbers match the hand calculation. They compute for the positive class by default (`pos_label=1`).
**3. Example to quote:**
> "`recall_score(y_true, y_pred)` returns 0.5, the same as my hand calculation."

**Seen in:** `main.py`, sklearn print

## zip()  `(Python built-in)`
**1. Definition:** `zip(a, b)` pairs items from two iterables position by position.
**2. In plain words:** Lets you walk true labels and predictions together: `for t, p in zip(y_true, y_pred)`.
**3. Example to quote:**
> "`zip(y_true, y_pred)` gives me each truth/prediction pair so I can count TP, FP, FN, TN."

**Seen in:** `main.py`, counting lines

## Counting with sum() and a condition  `(syntax)`
**1. Definition:** `sum(1 for x in items if cond)` counts how many items satisfy a condition.
**2. In plain words:** Each match contributes 1. It is how `tp`, `fp`, `fn` and `tn` are computed.
**3. Example to quote:**
> "`sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)` counts true positives."

**Seen in:** `main.py`, `tp`

## List repetition and concatenation  `(syntax)`
**1. Definition:** `[0] * 99` repeats a list; `[1] + [0] * 99` joins lists, producing 100 items.
**2. In plain words:** A quick way to build the imbalanced fraud example: one fraud and 99 normal.
**3. Example to quote:**
> "`[1] + [0] * 99` creates one fraud label followed by 99 normal ones."

**Seen in:** `main.py`, `y_true2`

## ZeroDivisionError / zero_division  `(concept)`
**1. Definition:** Computing a metric such as precision with no predicted positives divides by zero; plain Python raises `ZeroDivisionError`, while scikit-learn warns and returns 0 (controllable via `zero_division`).
**2. In plain words:** Handle the "model predicts no positives" case explicitly in your own code, or use scikit-learn's guard.
**3. Example to quote:**
> "If the model flags nothing, `tp / (tp + fp)` is `0/0`. sklearn returns 0.0 with a warning instead of crashing."

**Seen in:** `main.py`, BREAK IT comment

## f-string formatting (`:.2f`)  `(syntax)`
**1. Definition:** `{value:.2f}` formats a number with 2 decimal places inside an f-string.
**2. In plain words:** Prints `0.67` instead of `0.6666666666666666`.
**3. Example to quote:**
> "`f"{precision:.2f}"` prints 0.67."

**Seen in:** `main.py`, metric prints
