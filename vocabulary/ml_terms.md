# Machine Learning Terms

194 terms, alphabetical. Back to [index](index.md).

- [Accuracy](#accuracy)
- [Activation function](#activation-function)
- [Adam and AdamW](#adam-and-adamw)
- [Adam bias correction](#adam-bias-correction)
- [Adaptive (local) threshold](#adaptive-local-threshold)
- [Alias](#alias)
- [All-gather](#all-gather)
- [All-reduce](#all-reduce)
- [All-to-all](#all-to-all)
- [Arithmetic intensity](#arithmetic-intensity)
- [Autograd](#autograd)
- [Backpropagation](#backpropagation)
- [Backward pass](#backward-pass)
- [Batch normalization](#batch-normalization)
- [Batch size](#batch-size)
- [Batch size (training)](#batch-size-training)
- [Bias (parameter)](#bias-parameter)
- [Binarisation (thresholding)](#binarisation-thresholding)
- [Binary cross-entropy (log loss)](#binary-cross-entropy-log-loss)
- [Blank symbol (CTC)](#blank-symbol-ctc)
- [Bootstrap confidence interval](#bootstrap-confidence-interval)
- [Bounding box (bbox)](#bounding-box-bbox)
- [BPTT (backpropagation through time)](#bptt-backpropagation-through-time)
- [Bradley-Terry model](#bradley-terry-model)
- [Break-even volume](#break-even-volume)
- [Broadcasting](#broadcasting)
- [Catastrophic forgetting](#catastrophic-forgetting)
- [CER and WER](#cer-and-wer)
- [Chain rule (backpropagation through layers)](#chain-rule-backpropagation-through-layers)
- [Checkpoint](#checkpoint)
- [CNN character classifier](#cnn-character-classifier)
- [Column-parallel linear](#column-parallel-linear)
- [Computational graph](#computational-graph)
- [Concept drift](#concept-drift)
- [Confidence](#confidence)
- [Confusion matrix](#confusion-matrix)
- [Connected components](#connected-components)
- [Content-hash id](#content-hash-id)
- [Context precision (chunk attribution)](#context-precision-chunk-attribution)
- [Convolution (cross-correlation)](#convolution-cross-correlation)
- [Cosine similarity](#cosine-similarity)
- [CRNN (convolutional recurrent neural network)](#crnn-convolutional-recurrent-neural-network)
- [Cross-entropy loss](#cross-entropy-loss)
- [CTC (connectionist temporal classification)](#ctc-connectionist-temporal-classification)
- [Data augmentation](#data-augmentation)
- [Data collection](#data-collection)
- [Data drift](#data-drift)
- [Data leakage](#data-leakage)
- [Data parallelism (DP)](#data-parallelism-dp)
- [Data type (dtype)](#data-type-dtype)
- [DataLoader](#dataloader)
- [Decoupled weight decay (AdamW vs L2)](#decoupled-weight-decay-adamw-vs-l2)
- [Deskew (skew correction)](#deskew-skew-correction)
- [Dropout](#dropout)
- [Early stopping](#early-stopping)
- [Elo rating](#elo-rating)
- [Entropy (of a distribution)](#entropy-of-a-distribution)
- [Epoch](#epoch)
- [eval mode and no_grad](#eval-mode-and-no_grad)
- [Evaluation metric](#evaluation-metric)
- [Exact match and token F1](#exact-match-and-token-f1)
- [Exploding gradient](#exploding-gradient)
- [F1 score](#f1-score)
- [Feature](#feature)
- [Feature engineering](#feature-engineering)
- [Feature scaling](#feature-scaling)
- [Field-level accuracy](#field-level-accuracy)
- [fit vs transform](#fit-vs-transform)
- [Fixed-size bottleneck](#fixed-size-bottleneck)
- [Floating-point formats (fp32, fp16, bf16)](#floating-point-formats-fp32-fp16-bf16)
- [GAE (generalized advantage estimation)](#gae-generalized-advantage-estimation)
- [Gated MLP (GLU family)](#gated-mlp-glu-family)
- [GELU and SiLU (Swish)](#gelu-and-silu-swish)
- [Gradient](#gradient)
- [Gradient accumulation](#gradient-accumulation)
- [Gradient accumulation (PyTorch default)](#gradient-accumulation-pytorch-default)
- [Gradient check](#gradient-check)
- [Gradient clipping](#gradient-clipping)
- [Gradient descent](#gradient-descent)
- [He initialisation](#he-initialisation)
- [Hidden layer (MLP)](#hidden-layer-mlp)
- [hit@k, MRR, recall@k (retrieval metrics)](#hitk-mrr-recallk-retrieval-metrics)
- [Hyperparameter](#hyperparameter)
- [Idempotent upsert](#idempotent-upsert)
- [Incremental re-indexing (manifest)](#incremental-re-indexing-manifest)
- [Index version](#index-version)
- [Inference mode (no_grad)](#inference-mode-no_grad)
- [Inference pipeline](#inference-pipeline)
- [Iteration (step)](#iteration-step)
- [KL divergence](#kl-divergence)
- [L2 regularization](#l2-regularization)
- [Label](#label)
- [Label masking (-100)](#label-masking--100)
- [Label smoothing](#label-smoothing)
- [Layer normalization](#layer-normalization)
- [Layout analysis (page segmentation)](#layout-analysis-page-segmentation)
- [Lazy loading](#lazy-loading)
- [Learning rate](#learning-rate)
- [Learning rate (training)](#learning-rate-training)
- [Learning-rate schedule (warm-up and cosine decay)](#learning-rate-schedule-warm-up-and-cosine-decay)
- [Learning-rate scheduler (warm-up)](#learning-rate-scheduler-warm-up)
- [Levenshtein distance](#levenshtein-distance)
- [Linear separability](#linear-separability)
- [Log loss](#log-loss)
- [Log-odds (logit) and odds](#log-odds-logit-and-odds)
- [Logistic regression (logit model)](#logistic-regression-logit-model)
- [Logits](#logits)
- [Loss function](#loss-function)
- [Loss scaling](#loss-scaling)
- [Low-rank update](#low-rank-update)
- [LSTM (long short-term memory)](#lstm-long-short-term-memory)
- [Machine learning](#machine-learning)
- [Median filter](#median-filter)
- [Memory arithmetic (bytes per parameter)](#memory-arithmetic-bytes-per-parameter)
- [Memory-bound vs compute-bound](#memory-bound-vs-compute-bound)
- [Micro-batch](#micro-batch)
- [Mixed precision](#mixed-precision)
- [Mixed precision (autocast)](#mixed-precision-autocast)
- [ML pipeline](#ml-pipeline)
- [Model artifact](#model-artifact)
- [Model monitoring](#model-monitoring)
- [Model serialization (joblib)](#model-serialization-joblib)
- [Momentum (SGD with momentum)](#momentum-sgd-with-momentum)
- [Morphological operations (erosion, dilation, opening, closing)](#morphological-operations-erosion-dilation-opening-closing)
- [MSE](#mse)
- [NaN debugging](#nan-debugging)
- [nn.Module](#nnmodule)
- [Numerical gradient check](#numerical-gradient-check)
- [OCR (optical character recognition)](#ocr-optical-character-recognition)
- [Optimizer](#optimizer)
- [Optimizer (AdamW)](#optimizer-adamw)
- [Otsu's method](#otsus-method)
- [Overfitting](#overfitting)
- [Overfitting (training)](#overfitting-training)
- [Parameter (model)](#parameter-model)
- [Parameter (weight)](#parameter-weight)
- [pass@k](#passk)
- [Percentile latency (p50, p95)](#percentile-latency-p50-p95)
- [Percentiles p50/p95/p99](#percentiles-p50p95p99)
- [Perceptron (artificial neuron)](#perceptron-artificial-neuron)
- [Pipeline (sklearn)](#pipeline-sklearn)
- [Pipeline bubble](#pipeline-bubble)
- [Postprocessing](#postprocessing)
- [PPO (proximal policy optimization)](#ppo-proximal-policy-optimization)
- [Precision](#precision)
- [Prefix beam search](#prefix-beam-search)
- [Preprocessing](#preprocessing)
- [Probability calibration](#probability-calibration)
- [Projection profile](#projection-profile)
- [PSI](#psi)
- [Reading order](#reading-order)
- [Recall](#recall)
- [Receptive field](#receptive-field)
- [Recursive XY-cut](#recursive-xy-cut)
- [Rejection-sampling acceptance rule](#rejection-sampling-acceptance-rule)
- [ReLU and dead ReLU](#relu-and-dead-relu)
- [Reproducibility](#reproducibility)
- [Residual connection (skip connection)](#residual-connection-skip-connection)
- [Ridge point (machine balance)](#ridge-point-machine-balance)
- [RMSNorm](#rmsnorm)
- [RMSProp](#rmsprop)
- [RNN (recurrent neural network)](#rnn-recurrent-neural-network)
- [Roofline model](#roofline-model)
- [Row-parallel linear](#row-parallel-linear)
- [safetensors](#safetensors)
- [Seed (random seed)](#seed-random-seed)
- [SGD (stochastic gradient descent)](#sgd-stochastic-gradient-descent)
- [Sigmoid saturation](#sigmoid-saturation)
- [Softmax](#softmax)
- [state_dict](#state_dict)
- [Straight-through estimator (STE)](#straight-through-estimator-ste)
- [Supervised learning](#supervised-learning)
- [Table structure recognition](#table-structure-recognition)
- [TEDS (tree-edit-distance-based similarity)](#teds-tree-edit-distance-based-similarity)
- [Temperature (softmax)](#temperature-softmax)
- [Tensor](#tensor)
- [TensorDataset](#tensordataset)
- [TF-IDF](#tf-idf)
- [Train/test split](#traintest-split)
- [Train/validation split](#trainvalidation-split)
- [Trainable parameters](#trainable-parameters)
- [Training loop](#training-loop)
- [Training pipeline](#training-pipeline)
- [Training/serving skew](#trainingserving-skew)
- [Underfitting](#underfitting)
- [Utilisation](#utilisation)
- [Validation set](#validation-set)
- [Vanishing gradient](#vanishing-gradient)
- [Vectorizer](#vectorizer)
- [Weight](#weight)
- [Weight decay](#weight-decay)
- [Xavier (Glorot) initialisation](#xavier-glorot-initialisation)
- [XOR problem](#xor-problem)
- [ZeRO / FSDP](#zero-fsdp)

---

### Accuracy
*From module: 03_ml_foundations*

- **Technical definition:** Fraction of predictions that are correct.
- **Plain English:** How often the model is right.
- **Example / analogy:** Marks out of 100.
- **Snippet:**
```python
accuracy_score(y_true, y_pred)
```
- **Snippet explained:** Correct predictions / total.
- **Related terms:** [Precision](ml_terms.md#precision), [Recall](ml_terms.md#recall)

---

### Activation function
*From module: 15_neural_networks*

- **Technical definition:** A nonlinear function f applied elementwise after a linear map; without it a stack of linear layers collapses to one linear layer.
- **Plain English:** The bend that lets a network draw curves.
- **Example / analogy:** ReLU max(0, z), GELU, SiLU, tanh, sigmoid.
- **Snippet:**
```python
a = np.maximum(z, 0)
```
- **Snippet explained:** ReLU zeroes negatives and passes positives unchanged.
- **Related terms:** [ReLU and dead ReLU](ml_terms.md#relu-and-dead-relu), [Sigmoid saturation](ml_terms.md#sigmoid-saturation), [GELU and SiLU (Swish)](ml_terms.md#gelu-and-silu-swish)

---

### Adam and AdamW
*From module: 05_llm_parameters*

- **Technical definition:** Optimizers that track running averages of gradients and squared gradients to give each parameter its own adaptive step size; AdamW applies weight decay separately (decoupled).
- **Plain English:** A hiker with a personal pace per leg of the journey.
- **Example / analogy:** The default for LLM training; stores two extra numbers per parameter, so optimizer memory is about 2x the weights.
- **Snippet:**
```python
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
```
- **Snippet explained:** weight_decay shrinks weights slightly each step to curb overfitting.
- **Related terms:** [Optimizer](ml_terms.md#optimizer), [Weight decay](ml_terms.md#weight-decay), [Learning rate](ml_terms.md#learning-rate)

---

### Adam bias correction
*From module: 15_neural_networks*

- **Technical definition:** Adam's m and v start at 0 so they are biased low early; dividing by (1 - beta^t) corrects it, making the first step about lr in size.
- **Plain English:** A fix for a cold start of the running averages.
- **Example / analogy:** Without it the first step is about 3.16x lr (beta1 0.9, beta2 0.999).
- **Snippet:**
```python
mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
```
- **Snippet explained:** Corrected moments used in the update.
- **Related terms:** [Momentum (SGD with momentum)](ml_terms.md#momentum-sgd-with-momentum), [RMSProp](ml_terms.md#rmsprop), [Decoupled weight decay (AdamW vs L2)](ml_terms.md#decoupled-weight-decay-adamw-vs-l2)

---

### Adaptive (local) threshold
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** A threshold computed per pixel from its neighbourhood (for example mean minus an offset).
- **Plain English:** Compare each pixel with its neighbours, so shadows cancel.
- **Example / analogy:** Local mean (31x31) scored F1 0.949 on the shadowed page where Otsu scored 0.194.
- **Snippet:**
```python
ink = gray < local_mean - 25
```
- **Snippet explained:** Pixels darker than their neighbourhood by 25 levels count as ink.
- **Related terms:** [Otsu's method](ml_terms.md#otsus-method), [Binarisation (thresholding)](ml_terms.md#binarisation-thresholding)

---

### Alias
*From module: 99_extra_linked_terms*

- **Technical definition:** A movable name (such as `latest` or `production`) that points to a specific model version in a registry.
- **Plain English:** Callers ask for "production" and you decide which version that means; promoting or rolling back is just moving the alias.
- **Example / analogy:** A contact named "Mum" pointing to a phone number you can update.
- **Snippet:**
```python
aliases = {"production": "v3"}
version = aliases["production"]
```
- **Snippet explained:** Changing the dictionary value re-points "production" without changing callers.
- **Related terms:** [Model registry](api_and_deployment_terms.md#model-registry), [Model versioning](api_and_deployment_terms.md#model-versioning)

---

### All-gather
*From module: 21_llm_inference_engineering*

- **Technical definition:** A collective where every device receives the concatenation of all devices' slices.
- **Plain English:** Everyone ends up with the whole picture.
- **Example / analogy:** Each person holds one puzzle piece; afterwards everyone holds the whole puzzle.
- **Snippet:**
```python
dist.all_gather(out_list, t)
```
- **Snippet explained:** Fills out_list with each rank's tensor on every rank.
- **Related terms:** [All-reduce](ml_terms.md#all-reduce), [Column-parallel linear](ml_terms.md#column-parallel-linear)

---

### All-reduce
*From module: 21_llm_inference_engineering*

- **Technical definition:** A collective that sums a tensor held in partial form on every device and leaves the full sum on every device.
- **Plain English:** Everyone contributes a piece and everyone gets the total.
- **Example / analogy:** A group adds up their bills and each person receives the grand total.
- **Snippet:**
```python
dist.all_reduce(t, op=dist.ReduceOp.SUM)
```
- **Snippet explained:** torch.distributed call that sums t across ranks in place.
- **Related terms:** [All-gather](ml_terms.md#all-gather), [All-to-all](ml_terms.md#all-to-all)

---

### All-to-all
*From module: 21_llm_inference_engineering*

- **Technical definition:** A collective where each device sends a different chunk to every other device; used to dispatch tokens to experts in MoE.
- **Plain English:** Everyone swaps a different piece with everyone else.
- **Example / analogy:** Students each hand every other student a different card.
- **Snippet:**
```python
dist.all_to_all_single(out, inp)
```
- **Snippet explained:** Redistributes chunks between ranks.
- **Related terms:** [Expert parallelism (EP)](llm_terms.md#expert-parallelism-ep), [All-reduce](ml_terms.md#all-reduce)

---

### Arithmetic intensity
*From module: 21_llm_inference_engineering*

- **Technical definition:** FLOPs performed per byte moved from memory; the x-axis of the roofline.
- **Plain English:** Math done per byte fetched.
- **Example / analogy:** Decode intensity is about 2B/bytes_per_param.
- **Snippet:**
```python
intensity = 2 * batch / bytes_per_param
```
- **Snippet explained:** fp16 weights give intensity equal to the batch size.
- **Related terms:** [Roofline model](ml_terms.md#roofline-model), [Ridge point (machine balance)](ml_terms.md#ridge-point-machine-balance)

---

### Autograd
*From module: 03_ml_foundations*

- **Technical definition:** PyTorch's automatic differentiation: records tensor operations and computes gradients via backward().
- **Plain English:** PyTorch works out the derivatives for you.
- **Example / analogy:** A calculator that remembers every step so it can run them backwards.
- **Snippet:**
```python
w = torch.tensor(0., requires_grad=True)
loss = (w * 2 - 4) ** 2
loss.backward()
print(w.grad)
```
- **Snippet explained:** backward() fills w.grad with dloss/dw.
- **Related terms:** [Gradient](ml_terms.md#gradient), [Tensor](ml_terms.md#tensor), [Backpropagation](ml_terms.md#backpropagation)

---

### Backpropagation
*From module: 03_ml_foundations*

- **Technical definition:** Applying the chain rule backward through a network to compute gradients of the loss for every parameter.
- **Plain English:** How the blame for the error is passed back to each weight.
- **Example / analogy:** Tracing an error back through every step of a production line.
- **Snippet:**
```python
loss.backward()
```
- **Snippet explained:** Triggers backpropagation through the computation graph.
- **Related terms:** [Autograd](ml_terms.md#autograd), [Gradient](ml_terms.md#gradient)

---

### Backward pass
*From module: 06_llm_training_and_lora*

- **Technical definition:** Backpropagation: computing the gradient of the loss with respect to every parameter via the chain rule.
- **Plain English:** Working out which weights caused the error and by how much.
- **Example / analogy:** Tracing back through a recipe to find which ingredient made the dish too salty.
- **Snippet:**
```python
loss.backward()
```
- **Snippet explained:** Fills each parameter's .grad with d(loss)/d(parameter).
- **Related terms:** [Training loop](ml_terms.md#training-loop), [Gradient accumulation (PyTorch default)](ml_terms.md#gradient-accumulation-pytorch-default)

---

### Batch normalization
*From module: 15_neural_networks*

- **Technical definition:** Normalise each feature over the batch to zero mean and unit variance, then scale and shift with learned gamma and beta; uses running statistics at eval time.
- **Plain English:** Standardise each feature using the other examples in the batch.
- **Example / analogy:** Standard in CNNs; awkward with tiny or variable batches.
- **Snippet:**
```python
y = (x - x.mean(0)) / torch.sqrt(x.var(0, unbiased=False) + eps) * gamma + beta
```
- **Snippet explained:** Per-feature statistics over axis 0.
- **Related terms:** [Layer normalization](ml_terms.md#layer-normalization), [RMSNorm](ml_terms.md#rmsnorm), [Train/eval mode](python_syntax.md#traineval-mode)

---

### Batch size
*From module: 03_ml_foundations*

- **Technical definition:** Number of samples processed before each parameter update.
- **Plain English:** How many examples you look at before adjusting.
- **Example / analogy:** Grading papers in stacks of 32 before giving feedback.
- **Snippet:**
```python
DataLoader(train_set, batch_size=32, shuffle=True)
```
- **Snippet explained:** Loader yields 32 samples at a time.
- **Related terms:** [Epoch](ml_terms.md#epoch), [DataLoader](ml_terms.md#dataloader), [Hyperparameter](ml_terms.md#hyperparameter)

---

### Batch size (training)
*From module: 06_llm_training_and_lora*

- **Technical definition:** Number of examples processed per optimizer step.
- **Plain English:** How many examples the model looks at before updating itself once.
- **Example / analogy:** Marking 4 homework papers before adjusting how you teach.
- **Snippet:**
```python
loader = DataLoader(ds, batch_size=4)
```
- **Snippet explained:** Larger batches give smoother gradients but use more memory.
- **Related terms:** [Epoch](ml_terms.md#epoch), [Learning rate (training)](ml_terms.md#learning-rate-training), [DataLoader](ml_terms.md#dataloader)

---

### Bias (parameter)
*From module: 03_ml_foundations*

- **Technical definition:** A learned constant added to a weighted sum, letting the model shift its output.
- **Plain English:** The baseline output when inputs are zero.
- **Example / analogy:** A starting offset.
- **Snippet:**
```python
y_hat = w * x + b
```
- **Snippet explained:** b shifts the line up/down.
- **Related terms:** [Weight](ml_terms.md#weight)

---

### Binarisation (thresholding)
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** Turning a gray image into a black-and-white ink mask by comparing each pixel with a global or local threshold.
- **Plain English:** Deciding which pixels are ink and which are paper.
- **Example / analogy:** Otsu picks one global threshold; adaptive methods pick one per neighbourhood.
- **Snippet:**
```python
ink = gray <= otsu_threshold(gray)
```
- **Snippet explained:** True marks ink pixels.
- **Related terms:** [Otsu's method](ml_terms.md#otsus-method), [Adaptive (local) threshold](ml_terms.md#adaptive-local-threshold), [OCR (optical character recognition)](ml_terms.md#ocr-optical-character-recognition)

---

### Binary cross-entropy (log loss)
*From module: 15_neural_networks*

- **Technical definition:** L = -[y ln p + (1-y) ln(1-p)] averaged over samples; the negative log-likelihood of a Bernoulli model. With a sigmoid output its gradient w.r.t. the logit is p - y.
- **Plain English:** A penalty that explodes when the model is confident and wrong.
- **Example / analogy:** Predicting p=0.01 for a true 1 costs 4.6; predicting 0.5 costs 0.69.
- **Snippet:**
```python
-np.mean(y*np.log(p) + (1-y)*np.log(1-p))
```
- **Snippet explained:** Mean negative log probability assigned to the true label.
- **Related terms:** [Logistic regression (logit model)](ml_terms.md#logistic-regression-logit-model), [Probability calibration](ml_terms.md#probability-calibration)

---

### Blank symbol (CTC)
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** The extra output class in CTC meaning no character here; it separates repeated letters and absorbs padding frames.
- **Plain English:** A pause between letters.
- **Example / analogy:** a-a collapses to aa but aa collapses to a.
- **Snippet:**
```python
BLANK = 0
```
- **Snippet explained:** Index 0 is the blank by convention.
- **Related terms:** [CTC (connectionist temporal classification)](ml_terms.md#ctc-connectionist-temporal-classification)

---

### Bootstrap confidence interval
*From module: 17_finetuning_alignment_and_benchmarks*

- **Technical definition:** Resample evaluation items with replacement many times and take percentiles of the metric to get an interval.
- **Plain English:** Shake the test set to see how much the score wobbles.
- **Example / analogy:** 30 items at 50% give an interval about 0.35 wide.
- **Snippet:**
```python
np.quantile(means, [0.025, 0.975])
```
- **Snippet explained:** The percentile interval.
- **Related terms:** [pass@k](ml_terms.md#passk)

---

### Bounding box (bbox)
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** A rectangle [x0, y0, x1, y1] locating an element; the unit and origin vary by tool.
- **Plain English:** Where on the page.
- **Example / analogy:** Our contract: PDF points, origin top-left.
- **Snippet:**
```python
bbox = [round(v * 0.75, 1) for v in box_px]
```
- **Snippet explained:** Pixels to points at 96 dpi.
- **Related terms:** [PDF coordinate system](api_and_deployment_terms.md#pdf-coordinate-system), [Parsed document (layout elements)](llm_terms.md#parsed-document-layout-elements)

---

### BPTT (backpropagation through time)
*From module: 15_neural_networks*

- **Technical definition:** Backprop through the unrolled recurrence; the gradient is a product of Jacobians over T steps, so it vanishes or explodes exponentially.
- **Plain English:** Backprop where every time step is a layer.
- **Example / analogy:** Spectral radius below 1 vanishes, above 1 explodes.
- **Snippet:**
```python
loss.backward()   # through the unrolled loop
```
- **Snippet explained:** Autograd unrolls the Python loop.
- **Related terms:** [Vanishing gradient](ml_terms.md#vanishing-gradient), [Exploding gradient](ml_terms.md#exploding-gradient), [LSTM (long short-term memory)](ml_terms.md#lstm-long-short-term-memory)

---

### Bradley-Terry model
*From module: 17_finetuning_alignment_and_benchmarks*

- **Technical definition:** A model of pairwise comparisons where P(i beats j) = sigmoid(s_i - s_j) for latent scores s.
- **Plain English:** Turns 'A beat B' votes into numeric strengths.
- **Example / analogy:** Used for reward-model training and for Arena rankings.
- **Snippet:**
```python
loss = -F.logsigmoid(r_chosen - r_rejected).mean()
```
- **Snippet explained:** Negative log-likelihood of the observed preference.
- **Related terms:** [Reward model](llm_terms.md#reward-model), [Elo rating](ml_terms.md#elo-rating)

---

### Break-even volume
*From module: 21_llm_inference_engineering*

- **Technical definition:** Monthly token volume above which fixed-cost self-hosting is cheaper than a pay-per-token API: fixed monthly cost divided by the API's blended price per token.
- **Plain English:** The point where buying beats renting.
- **Example / analogy:** Buying a car versus taking taxis.
- **Snippet:**
```python
be = n_gpu * price_h * 730 / blended_price_per_mtok
```
- **Snippet explained:** Gives millions of tokens per month.
- **Related terms:** [Cost per million tokens](llm_terms.md#cost-per-million-tokens), [Utilisation](ml_terms.md#utilisation)

---

### Broadcasting
*From module: 03_ml_foundations*

- **Technical definition:** Numpy/PyTorch rule that expands smaller arrays across larger ones during element-wise operations.
- **Plain English:** Automatic repetition so shapes line up.
- **Example / analogy:** Adding the same tax to every price in a list.
- **Snippet:**
```python
(X - X.mean(axis=0)) / X.std(axis=0)
```
- **Snippet explained:** A (n,2) array minus a (2,) vector subtracts per column.
- **Related terms:** [Tensor](ml_terms.md#tensor), [Feature scaling](ml_terms.md#feature-scaling)

---

### Catastrophic forgetting
*From module: 17_finetuning_alignment_and_benchmarks*

- **Technical definition:** Loss of previously learned abilities after fine-tuning on a narrow task.
- **Plain English:** Learning a new trick and forgetting old ones.
- **Example / analogy:** Measured by regression probes (general MCQ, general-text perplexity) before and after.
- **Snippet:**
```python
perplexity(model, tok, general_text)
```
- **Snippet explained:** Probe of general language ability.
- **Related terms:** [Perplexity](llm_terms.md#perplexity), [PEFT (parameter-efficient fine-tuning)](llm_terms.md#peft-parameter-efficient-fine-tuning)

---

### CER and WER
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** Character and word error rate: (substitutions + deletions + insertions) divided by reference length in characters or words.
- **Plain English:** How many fixes the text needs, per character or per word.
- **Example / analogy:** CER 3.4% was WER 15.4% in the module run.
- **Snippet:**
```python
cer = lev(ref, hyp) / len(ref)
```
- **Snippet explained:** Edit distance divided by reference length.
- **Related terms:** [Levenshtein distance](ml_terms.md#levenshtein-distance), [Field-level accuracy](ml_terms.md#field-level-accuracy)

---

### Chain rule (backpropagation through layers)
*From module: 15_neural_networks*

- **Technical definition:** dL/dx = dL/dy * dy/dx applied layer by layer from the loss backwards, reusing cached forward values; a weight gradient has the weight's shape, dW = input^T @ dout.
- **Plain English:** Blame flows backward, multiplied by each layer's local slope.
- **Example / analogy:** For softmax cross-entropy the logit gradient is (p - onehot)/N.
- **Snippet:**
```python
dW2 = a1.T @ dout; da1 = dout @ W2.T; dz1 = da1 * (1 - a1**2)
```
- **Snippet explained:** Three lines of backward for a tanh hidden layer.
- **Related terms:** [Computational graph](ml_terms.md#computational-graph), [Gradient check](ml_terms.md#gradient-check)

---

### Checkpoint
*From module: 06_llm_training_and_lora*

- **Technical definition:** A saved snapshot of model weights (and config/tokenizer) that can be reloaded later.
- **Plain English:** A save-game for your model.
- **Example / analogy:** Saving a video game before a boss fight.
- **Snippet:**
```python
model.save_pretrained('ckpt'); tok.save_pretrained('ckpt')
```
- **Snippet explained:** Writes config + safetensors weights, plus tokenizer files to the folder.
- **Related terms:** [safetensors](ml_terms.md#safetensors), [state_dict](ml_terms.md#state_dict)

---

### CNN character classifier
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** A small convolutional network mapping a cropped character image to one of the character classes.
- **Plain English:** Learns what an a looks like despite font and noise.
- **Example / analogy:** With augmentation a small CNN reached 0.782 on an unseen font plus rotation plus noise.
- **Snippet:**
```python
logits = cnn(x)  # (N, 62)
```
- **Snippet explained:** One score per character class.
- **Related terms:** [CRNN (convolutional recurrent neural network)](ml_terms.md#crnn-convolutional-recurrent-neural-network), [Data augmentation](ml_terms.md#data-augmentation)

---

### Column-parallel linear
*From module: 21_llm_inference_engineering*

- **Technical definition:** Splitting a weight matrix W by output columns so each device computes a slice of Y = XW from the full X; slices are concatenated.
- **Plain English:** Each GPU produces some of the output columns.
- **Example / analogy:** Each cook makes some of the dishes from the same ingredients.
- **Snippet:**
```python
Y = np.concatenate([X @ w for w in np.split(W, N, axis=1)], axis=1)
```
- **Snippet explained:** Concatenating per-device outputs equals the unsplit result.
- **Related terms:** [Row-parallel linear](ml_terms.md#row-parallel-linear), [All-gather](ml_terms.md#all-gather), [Megatron MLP pattern](llm_terms.md#megatron-mlp-pattern)

---

### Computational graph
*From module: 15_neural_networks*

- **Technical definition:** A directed acyclic graph whose nodes are operations/tensors, recorded during the forward pass so backward can traverse it in reverse topological order.
- **Plain English:** The receipt of every calculation, replayed backwards to assign blame.
- **Example / analogy:** PyTorch builds it on the fly (define-by-run) and frees it after backward().
- **Snippet:**
```python
c = a * b + a   # a has two outgoing edges
```
- **Snippet explained:** Node a is used twice so its gradient is the sum of two paths.
- **Related terms:** [Chain rule (backpropagation through layers)](ml_terms.md#chain-rule-backpropagation-through-layers)

---

### Concept drift
*From module: 10_ml_system_architecture*

- **Technical definition:** A change in the relationship between inputs and the correct output over time (new slang, new fraud tactics) that needs fresh labels to detect.
- **Plain English:** The right answer for the same input has changed.
- **Example / analogy:** 'Sick' now meaning 'great'.
- **Snippet:**
n/a
- **Snippet explained:** n/a
- **Related terms:** [Data drift](ml_terms.md#data-drift), [Model monitoring](ml_terms.md#model-monitoring)

---

### Confidence
*From module: 10_ml_system_architecture*

- **Technical definition:** The probability a model assigns to its predicted class (e.g. max of predict_proba); not guaranteed to be calibrated.
- **Plain English:** How sure the model says it is.
- **Example / analogy:** A weather forecast's '70% chance'.
- **Snippet:**
```python
probs = model.predict_proba([text])[0]
confidence = probs.max()
```
- **Snippet explained:** Maximum class probability.
- **Related terms:** [Postprocessing](ml_terms.md#postprocessing), [Log loss](ml_terms.md#log-loss)

---

### Confusion matrix
*From module: 03_ml_foundations*

- **Technical definition:** Table counting true/false positives and negatives.
- **Plain English:** A scoreboard of right and wrong guesses by class.
- **Example / analogy:** A 2x2 grid of hits and misses.
- **Snippet:**
```python
confusion_matrix(y_true, y_pred)   # rows: truth; cols: prediction
```
- **Snippet explained:** Rows are truth, columns predictions.
- **Related terms:** [Precision](ml_terms.md#precision), [Recall](ml_terms.md#recall)

---

### Connected components
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** Labelling groups of touching foreground pixels; each group is one blob, typically a character.
- **Plain English:** Finding separate pieces of ink.
- **Example / analogy:** An i is two components; rn touching is one.
- **Snippet:**
```python
lab, n = scipy.ndimage.label(line_ink)
```
- **Snippet explained:** Returns a label image and the number of components.
- **Related terms:** [Projection profile](ml_terms.md#projection-profile), [CTC (connectionist temporal classification)](ml_terms.md#ctc-connectionist-temporal-classification)

---

### Content-hash id
*From module: 20_llm_system_lld*

- **Technical definition:** An identifier computed by hashing the content (for chunks: document id, section and text), so identical content always gets an identical id.
- **Plain English:** A fingerprint rather than a ticket number.
- **Example / analogy:** Re-indexing an unchanged document produces the same chunk ids, so nothing is re-embedded.
- **Snippet:**
```python
content_hash(doc_id, section, text)  # 16 hex chars
```
- **Snippet explained:** Joins the parts with a separator and keeps the first 16 hex characters of sha256.
- **Related terms:** [Idempotent upsert](ml_terms.md#idempotent-upsert), [Index version](ml_terms.md#index-version)

---

### Context precision (chunk attribution)
*From module: 19_chunking_deep_dive*

- **Technical definition:** The share of the characters (or tokens) sent to the LLM that belong to the answer evidence. Low values mean you pay for text that does not help; attribution metrics also ask which chunk an answer sentence is supported by.
- **Plain English:** How much of what you sent was actually useful.
- **Example / analogy:** Mailing a 300-page book to deliver one sentence.
- **Snippet:**
```python
precision = gold_chars_in_context / total_context_chars
```
- **Snippet explained:** Computed from character offsets, so it works for any chunker.
- **Related terms:** [hit@k, MRR, recall@k (retrieval metrics)](ml_terms.md#hitk-mrr-recallk-retrieval-metrics)

---

### Convolution (cross-correlation)
*From module: 15_neural_networks*

- **Technical definition:** Slide a small kernel over the input computing local dot products with shared weights; output size floor((n + 2p - k)/s) + 1. Libraries implement cross-correlation (kernel not flipped).
- **Plain English:** One small pattern detector reused at every position.
- **Example / analogy:** 64 3x3 filters on RGB: 1,792 parameters versus 150 million for a dense layer.
- **Snippet:**
```python
y = F.conv2d(x, w, stride=1, padding=1)
```
- **Snippet explained:** Batch of images with (out, in, k, k) kernels.
- **Related terms:** [Receptive field](ml_terms.md#receptive-field)

---

### Cosine similarity
*From module: 04_llm_fundamentals*

- **Technical definition:** The cosine of the angle between two vectors: dot(a, b) / (|a| |b|); 1 = same direction, 0 = orthogonal, -1 = opposite.
- **Plain English:** How much two arrows point the same way, ignoring their length.
- **Example / analogy:** cosine('cat', 'dog') = 0.57 but cosine('cat', 'economics') = 0.20 in GPT-2 embeddings.
- **Snippet:**
```python
import torch.nn.functional as F
F.cosine_similarity(a, b, dim=0).item()
```
- **Snippet explained:** Compares two 1-D vectors and converts the 0-d result to a Python float.
- **Related terms:** [Embedding](llm_terms.md#embedding), [Position embedding](llm_terms.md#position-embedding)

---

### CRNN (convolutional recurrent neural network)
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** A CNN feature extractor, a recurrent (BiGRU/BiLSTM) sequence model and a CTC output layer that reads a whole text line.
- **Plain English:** Looks at the line as a sequence of vertical slices.
- **Example / analogy:** A 133k-parameter CRNN reached CER 0.122 on an unseen font.
- **Snippet:**
```python
feats = cnn(img); out, _ = gru(feats)
```
- **Snippet explained:** Convolutions then bidirectional GRU produce per-frame class scores.
- **Related terms:** [CTC (connectionist temporal classification)](ml_terms.md#ctc-connectionist-temporal-classification), [CNN character classifier](ml_terms.md#cnn-character-classifier)

---

### Cross-entropy loss
*From module: 03_ml_foundations*

- **Technical definition:** -log of the probability assigned to the correct class; standard classification and language-model loss.
- **Plain English:** Penalises the model heavily when it is confident and wrong.
- **Example / analogy:** A quiz where a confident wrong answer costs the most.
- **Snippet:**
```python
F.cross_entropy(logits, target)
```
- **Snippet explained:** Takes raw logits and class indices; applies softmax internally.
- **Related terms:** [Softmax](ml_terms.md#softmax), [Logits](ml_terms.md#logits), [Loss function](ml_terms.md#loss-function)

---

### CTC (connectionist temporal classification)
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** A loss and decoding scheme that sums over all frame-level alignments that collapse (merge repeats, drop blanks) to the target text.
- **Plain English:** Lets the network read without being told where each letter is.
- **Example / analogy:** There are 15 valid alignments of ab in 4 frames.
- **Snippet:**
```python
loss = nn.CTCLoss(blank=0)(log_probs, targets, in_len, tgt_len)
```
- **Snippet explained:** Computes -log of the summed path probabilities.
- **Related terms:** [CRNN (convolutional recurrent neural network)](ml_terms.md#crnn-convolutional-recurrent-neural-network), [Prefix beam search](ml_terms.md#prefix-beam-search), [Blank symbol (CTC)](ml_terms.md#blank-symbol-ctc)

---

### Data augmentation
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** Training on randomly altered copies of the data (rotation, noise, scale) to improve robustness.
- **Plain English:** Practising on messy versions.
- **Example / analogy:** Augmentation took held-out accuracy from 0.468 to 0.782.
- **Snippet:**
```python
x = rotate(x, rng.uniform(-8, 8)) + noise
```
- **Snippet explained:** Each epoch sees different distortions.
- **Related terms:** [CNN character classifier](ml_terms.md#cnn-character-classifier)

---

### Data collection
*From module: 10_ml_system_architecture*

- **Technical definition:** Gathering raw examples from files, databases, APIs or user feedback that represent the cases the model will face in production.
- **Plain English:** Getting the examples the model will learn from.
- **Example / analogy:** Gathering ingredients before cooking.
- **Snippet:**
```python
texts, labels = get_training_data(n=200, seed=0)
```
- **Snippet explained:** A deterministic generator stands in for real data sources.
- **Related terms:** [Preprocessing](ml_terms.md#preprocessing), [Data leakage](ml_terms.md#data-leakage)

---

### Data drift
*From module: 10_ml_system_architecture*

- **Technical definition:** A change in the distribution of model inputs between training and production, potentially lowering accuracy; detected with statistics such as PSI.
- **Plain English:** The kind of data coming in has changed.
- **Example / analogy:** A language model trained on formal text now seeing slang.
- **Snippet:**
```python
psi(reference_lengths, live_lengths) > 0.25
```
- **Snippet explained:** PSI above 0.25 signals significant drift.
- **Related terms:** [Concept drift](ml_terms.md#concept-drift), [PSI](ml_terms.md#psi)

---

### Data leakage
*From module: 03_ml_foundations*

- **Technical definition:** Information from validation/test data leaking into training or preprocessing, inflating metrics.
- **Plain English:** Peeking at the exam answers.
- **Example / analogy:** Fitting a scaler on all data before splitting.
- **Snippet:**
```python
pipe = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])
pipe.fit(X_train, y_train)
```
- **Snippet explained:** Pipeline fits the vectorizer on training data only.
- **Related terms:** [Train/validation split](ml_terms.md#trainvalidation-split), [Pipeline (sklearn)](ml_terms.md#pipeline-sklearn)

---

### Data parallelism (DP)
*From module: 21_llm_inference_engineering*

- **Technical definition:** Replicating the whole model on several devices and splitting the batch or the requests between them.
- **Plain English:** Many copies of the same model, each serving different inputs.
- **Example / analogy:** Opening more checkout lanes.
- **Snippet:**
```python
replicas = 4  # each serves 1/4 of requests
```
- **Snippet explained:** Scales throughput linearly with no communication when serving.
- **Related terms:** [ZeRO / FSDP](ml_terms.md#zero-fsdp), [Tensor parallelism (TP)](llm_terms.md#tensor-parallelism-tp)

---

### Data type (dtype)
*From module: 04_llm_fundamentals*

- **Technical definition:** How each number is stored: float32 (4 bytes), float16/bfloat16 (2), int8 (1), int4 (0.5). Fewer bytes means less memory but less precision.
- **Plain English:** How many decimal digits of precision you keep per number.
- **Example / analogy:** Casting a model with model.half() halves its memory.
- **Snippet:**
```python
p = next(model.parameters())
print(p.dtype, p.element_size())
```
- **Snippet explained:** Shows the dtype (float32) and its size in bytes (4).
- **Related terms:** [Parameter count](llm_terms.md#parameter-count), [Parameter (weight)](ml_terms.md#parameter-weight)

---

### DataLoader
*From module: 03_ml_foundations*

- **Technical definition:** PyTorch utility that batches, shuffles and iterates over a dataset.
- **Plain English:** Hands the model data in neat batches.
- **Example / analogy:** A cafeteria server giving trays of fixed size.
- **Snippet:**
```python
for xb, yb in DataLoader(ds, batch_size=32):
    ...
```
- **Snippet explained:** Each iteration yields one batch of inputs and labels.
- **Related terms:** [Batch size](ml_terms.md#batch-size), [Epoch](ml_terms.md#epoch)

---

### Decoupled weight decay (AdamW vs L2)
*From module: 15_neural_networks*

- **Technical definition:** AdamW shrinks weights directly, p *= (1 - lr wd), outside the adaptive scaling; L2 adds wd p to the gradient, which Adam then divides by sqrt(v), so the two differ under Adam (same under plain SGD).
- **Plain English:** Weight decay done properly for Adam.
- **Example / analogy:** Adam(wd=0.05) crushed weights to norm 1.8; AdamW(wd=0.05) left them near 29.7.
- **Snippet:**
```python
opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.1)
```
- **Snippet explained:** Decoupled decay; do not decay biases and norm gains.
- **Related terms:** [Adam bias correction](ml_terms.md#adam-bias-correction), [L2 regularization](ml_terms.md#l2-regularization), [Learning-rate schedule (warm-up and cosine decay)](ml_terms.md#learning-rate-schedule-warm-up-and-cosine-decay)

---

### Deskew (skew correction)
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** Estimating the rotation of a scanned page and rotating it back; commonly by maximising the variance of the row projection profile.
- **Plain English:** Straightening a crooked scan.
- **Example / analogy:** A 4 degree skew made row projection find 1 line instead of 4.
- **Snippet:**
```python
best = max(angles, key=lambda a: profile_variance(rotate(img, a)))
```
- **Snippet explained:** Tries angles and keeps the one with the sharpest row profile.
- **Related terms:** [Projection profile](ml_terms.md#projection-profile), [Layout analysis (page segmentation)](ml_terms.md#layout-analysis-page-segmentation)

---

### Dropout
*From module: 04_llm_fundamentals*

- **Technical definition:** A regularisation technique that randomly zeroes some activations during training to reduce overfitting; disabled at inference.
- **Plain English:** Making students work with random teammates absent so nobody relies on one person.
- **Example / analogy:** model.train() enables it; model.eval() disables it.
- **Snippet:**
```python
model.train()   # dropout active
model.eval()    # dropout off
```
- **Snippet explained:** Switches the mode flag for every layer in the model.
- **Related terms:** [eval mode and no_grad](ml_terms.md#eval-mode-and-no_grad), [Fine-tuning](llm_terms.md#fine-tuning)

---

### Early stopping
*From module: 03_ml_foundations*

- **Technical definition:** Stop training when validation loss stops improving to avoid overfitting.
- **Plain English:** Quit while you're ahead.
- **Example / analogy:** Stop studying when practice-test scores stop improving.
- **Snippet:**
```python
if val_loss > best_val_loss: patience_left -= 1
```
- **Snippet explained:** Count epochs without improvement and stop at zero.
- **Related terms:** [Overfitting](ml_terms.md#overfitting), [Epoch](ml_terms.md#epoch)

---

### Elo rating
*From module: 17_finetuning_alignment_and_benchmarks*

- **Technical definition:** Online rating update r_i += K*(score - expected) with expected = 1/(1+10^((r_j-r_i)/400)); depends on the order of games.
- **Plain English:** Chess-style rating after each match.
- **Example / analogy:** Bradley-Terry maximum likelihood is the order-independent alternative.
- **Snippet:**
```python
e = 1/(1+10**((rj-ri)/400))
```
- **Snippet explained:** Expected score of i.
- **Related terms:** [Bradley-Terry model](ml_terms.md#bradley-terry-model)

---

### Entropy (of a distribution)
*From module: 05_llm_parameters*

- **Technical definition:** -sum p log p: a measure of uncertainty; 0 for a certain outcome, log(N) for a uniform one over N options.
- **Plain English:** How surprised you expect to be by the next token.
- **Example / analogy:** Entropy was 0.01 at T=0.2 and 10.28 at T=3.0 (uniform over 50,257 would be 10.8).
- **Snippet:**
```python
H = -(p[p > 0] * p[p > 0].log()).sum()
```
- **Snippet explained:** Ignore zero probabilities (log 0 is undefined) and sum p*log p.
- **Related terms:** [Temperature (sampling)](llm_terms.md#temperature-sampling), [Perplexity](llm_terms.md#perplexity)

---

### Epoch
*From module: 03_ml_foundations*

- **Technical definition:** One complete pass through the entire training dataset.
- **Plain English:** One full read-through of the training data.
- **Example / analogy:** Reading the whole textbook once.
- **Snippet:**
```python
for epoch in range(10):
    ...
```
- **Snippet explained:** Ten passes over the data.
- **Related terms:** [Batch size](ml_terms.md#batch-size), [Training loop](ml_terms.md#training-loop)

---

### eval mode and no_grad
*From module: 04_llm_fundamentals*

- **Technical definition:** model.eval() turns off training-only behaviour (dropout); torch.no_grad() stops gradient tracking. They are separate and normally used together for inference.
- **Plain English:** 'Exam mode' for behaviour and 'don't take notes' for memory.
- **Example / analogy:** Forgetting no_grad on large inputs wastes memory storing a computation graph you never use.
- **Snippet:**
```python
model.eval()
with torch.no_grad():
    logits = model(ids).logits
```
- **Snippet explained:** eval() changes layer behaviour; no_grad() is a context manager that restores tracking on exit.
- **Related terms:** [Inference](llm_terms.md#inference), [Dropout](ml_terms.md#dropout), [Backpropagation](ml_terms.md#backpropagation)

---

### Evaluation metric
*From module: 10_ml_system_architecture*

- **Technical definition:** A number summarising model quality on held-out data (accuracy, precision/recall, F1, log loss...).
- **Plain English:** A score for how good the model is.
- **Example / analogy:** Exam marks.
- **Snippet:**
```python
accuracy_score(y_test, model.predict(X_test))
```
- **Snippet explained:** Fraction of correct predictions.
- **Related terms:** [Log loss](ml_terms.md#log-loss), [Train/test split](ml_terms.md#traintest-split)

---

### Exact match and token F1
*From module: 17_finetuning_alignment_and_benchmarks*

- **Technical definition:** EM: normalised prediction equals the gold string; F1: harmonic mean of token precision and recall after normalisation.
- **Plain English:** Strict vs partial credit for short answers.
- **Example / analogy:** 'Paris, France' vs 'Paris': EM 0, F1 0.67.
- **Snippet:**
```python
token_f1('Paris, France','Paris')
```
- **Snippet explained:** Partial overlap score.
- **Related terms:** [Answer extraction](llm_terms.md#answer-extraction)

---

### Exploding gradient
*From module: 15_neural_networks*

- **Technical definition:** Gradients grow exponentially with depth or time when per-layer factors exceed 1; produces huge updates and inf/NaN.
- **Plain English:** The learning signal gets louder at every layer until it blows up.
- **Example / analogy:** Recurrent nets with a recurrent matrix of spectral radius above 1.
- **Snippet:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```
- **Snippet explained:** Rescales the whole gradient so its norm is at most 1.
- **Related terms:** [Gradient clipping](ml_terms.md#gradient-clipping), [Vanishing gradient](ml_terms.md#vanishing-gradient), [BPTT (backpropagation through time)](ml_terms.md#bptt-backpropagation-through-time)

---

### F1 score
*From module: 03_ml_foundations*

- **Technical definition:** Harmonic mean of precision and recall: 2PR/(P+R).
- **Plain English:** One number balancing precision and recall.
- **Example / analogy:** A team grade that's low if either teammate is weak.
- **Snippet:**
```python
f1_score(y_true, y_pred)
```
- **Snippet explained:** Returns the harmonic mean.
- **Related terms:** [Precision](ml_terms.md#precision), [Recall](ml_terms.md#recall)

---

### Feature
*From module: 03_ml_foundations*

- **Technical definition:** An input variable used by a model to make a prediction.
- **Plain English:** A fact about each example that the model may use.
- **Example / analogy:** For house price: size, rooms, location.
- **Snippet:**
```python
X = np.array([ [50, 2], [80, 3] ])   # columns = features
```
- **Snippet explained:** Each row is a house; each column is a feature.
- **Related terms:** [Label](ml_terms.md#label), [Feature scaling](ml_terms.md#feature-scaling)

---

### Feature engineering
*From module: 10_ml_system_architecture*

- **Technical definition:** Turning raw data into numeric features a model can use (TF-IDF vectors, embeddings, scaled numbers, counts).
- **Plain English:** Converting text or records into numbers the model understands.
- **Example / analogy:** Measuring ingredients into grams.
- **Snippet:**
```python
vec = TfidfVectorizer()
X = vec.fit_transform(texts)
```
- **Snippet explained:** Learns the vocabulary and returns a texts-by-words matrix.
- **Related terms:** [TF-IDF](ml_terms.md#tf-idf), [fit vs transform](ml_terms.md#fit-vs-transform), [Preprocessing](ml_terms.md#preprocessing)

---

### Feature scaling
*From module: 03_ml_foundations*

- **Technical definition:** Rescaling features (e.g. standardisation) so they have comparable ranges, which speeds up training.
- **Plain English:** Putting all columns on a similar scale.
- **Example / analogy:** Converting height in cm and weight in kg into 'how many standard deviations from average'.
- **Snippet:**
```python
X_scaled = (X - X.mean(0)) / X.std(0)
```
- **Snippet explained:** Each column gets mean 0 and std 1.
- **Related terms:** [Feature](ml_terms.md#feature), [Broadcasting](ml_terms.md#broadcasting)

---

### Field-level accuracy
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** Fraction of extracted key-value fields (date, total, ID) that equal the gold values, under exact, normalised or fuzzy matching.
- **Plain English:** Did we get the numbers the business needs?
- **Example / analogy:** Fuzzy matching passed 2300.00 for 230.00; use exact for amounts.
- **Snippet:**
```python
ok = got['total'] == gold['total']
```
- **Snippet explained:** Exact string match for an amount.
- **Related terms:** [CER and WER](ml_terms.md#cer-and-wer), [TEDS (tree-edit-distance-based similarity)](ml_terms.md#teds-tree-edit-distance-based-similarity)

---

### fit vs transform
*From module: 10_ml_system_architecture*

- **Technical definition:** fit learns parameters from data (vocabulary, scaling statistics, weights); transform applies learned parameters; fit_transform does both on the same data. Only fit on training data.
- **Plain English:** fit = learn the rules, transform = apply the rules.
- **Example / analogy:** Studying (fit) then taking the exam with what you learned (transform).
- **Snippet:**
```python
Xtr = vec.fit_transform(train)
Xte = vec.transform(test)   # never fit on test
```
- **Snippet explained:** Test data is transformed with the training vocabulary.
- **Related terms:** [Data leakage](ml_terms.md#data-leakage), [Vectorizer](ml_terms.md#vectorizer)

---

### Fixed-size bottleneck
*From module: 15_neural_networks*

- **Technical definition:** An encoder-decoder RNN squeezes the whole input into one vector; accuracy falls as sequences lengthen (copying 24 tokens: 0.126, chance level). Attention lets the decoder read every position instead.
- **Plain English:** Summarising a book on a sticky note.
- **Example / analogy:** Motivation for attention in seq2seq.
- **Snippet:**
```python
ctx = enc_states[-1]   # one vector for everything
```
- **Snippet explained:** The last hidden state is all the decoder sees.
- **Related terms:** [LSTM (long short-term memory)](ml_terms.md#lstm-long-short-term-memory), [BPTT (backpropagation through time)](ml_terms.md#bptt-backpropagation-through-time)

---

### Floating-point formats (fp32, fp16, bf16)
*From module: 15_neural_networks*

- **Technical definition:** fp32: 8 exponent + 23 mantissa bits; fp16: 5 + 10, max 65,504; bf16: 8 + 7, fp32's range with coarser precision.
- **Plain English:** Different trade-offs between range and detail.
- **Example / analogy:** bf16 needs no loss scaling; fp16 does.
- **Snippet:**
```python
torch.finfo(torch.bfloat16).max
```
- **Snippet explained:** Range limit of bf16 (about 3.4e38).
- **Related terms:** [Loss scaling](ml_terms.md#loss-scaling), [Mixed precision (autocast)](ml_terms.md#mixed-precision-autocast), [NaN debugging](ml_terms.md#nan-debugging)

---

### GAE (generalized advantage estimation)
*From module: 17_finetuning_alignment_and_benchmarks*

- **Technical definition:** An advantage estimate A_t = sum (gamma*lambda)^l * delta_{t+l} with TD errors delta = r + gamma*V(s') - V(s).
- **Plain English:** A smoothed estimate of how much better an action was than expected.
- **Example / analogy:** Needs a learned value network.
- **Snippet:**
```python
delta = r[t] + gamma*v[t+1] - v[t]
```
- **Snippet explained:** One TD error.
- **Related terms:** [PPO (proximal policy optimization)](ml_terms.md#ppo-proximal-policy-optimization)

---

### Gated MLP (GLU family)
*From module: 15_neural_networks*

- **Technical definition:** A feed-forward block that multiplies two projections elementwise, down(act(x W_gate) * (x W_up)); SwiGLU uses SiLU as act. Hidden size is about 8/3 d to keep parameters equal to a 4d GELU MLP.
- **Plain English:** One path decides how much of the other path passes.
- **Example / analogy:** Llama's gate_proj, up_proj, down_proj.
- **Snippet:**
```python
h = torch.nn.functional.silu(x @ Wg) * (x @ Wu)
```
- **Snippet explained:** Gate times content, then project down.
- **Related terms:** [GELU and SiLU (Swish)](ml_terms.md#gelu-and-silu-swish), [Activation function](ml_terms.md#activation-function), [Hidden layer (MLP)](ml_terms.md#hidden-layer-mlp)

---

### GELU and SiLU (Swish)
*From module: 15_neural_networks*

- **Technical definition:** Smooth ReLU-like functions: GELU(x) = x Phi(x), SiLU(x) = x sigmoid(x) (Swish with beta 1).
- **Plain English:** ReLU with rounded corners.
- **Example / analogy:** GELU in BERT/GPT-2; SiLU in the gate of SwiGLU in Llama.
- **Snippet:**
```python
y = torch.nn.functional.silu(x)
```
- **Snippet explained:** Equals x * torch.sigmoid(x).
- **Related terms:** [Activation function](ml_terms.md#activation-function), [ReLU and dead ReLU](ml_terms.md#relu-and-dead-relu), [Gated MLP (GLU family)](ml_terms.md#gated-mlp-glu-family)

---

### Gradient
*From module: 03_ml_foundations*

- **Technical definition:** The vector of partial derivatives of the loss with respect to each parameter; points toward steepest increase.
- **Plain English:** Which way and how steeply the loss changes if you nudge each parameter.
- **Example / analogy:** The slope of a hill under your feet.
- **Snippet:**
```python
grad_w = 2 * np.mean(error * x)
```
- **Snippet explained:** Derivative of MSE with respect to w.
- **Related terms:** [Gradient descent](ml_terms.md#gradient-descent), [Autograd](ml_terms.md#autograd), [Backpropagation](ml_terms.md#backpropagation)

---

### Gradient accumulation
*From module: 05_llm_parameters*

- **Technical definition:** Summing gradients over several small batches before one optimizer step, emulating a larger batch with less memory.
- **Plain English:** Marking five small piles of homework before changing your study method once.
- **Example / analogy:** Effective batch 32 = batch 8 x 4 accumulation steps.
- **Snippet:**
```python
loss = loss / 4
loss.backward()
if (i + 1) % 4 == 0:
    opt.step(); opt.zero_grad()
```
- **Snippet explained:** Gradients add up in .grad until zero_grad is called after every 4th batch.
- **Related terms:** [Batch size](ml_terms.md#batch-size), [Iteration (step)](ml_terms.md#iteration-step)

---

### Gradient accumulation (PyTorch default)
*From module: 06_llm_training_and_lora*

- **Technical definition:** PyTorch adds new gradients to existing .grad values, so they must be cleared with zero_grad() each step.
- **Plain English:** Old corrections pile up unless you wipe them.
- **Example / analogy:** A notepad you must erase before the next calculation.
- **Snippet:**
```python
opt.zero_grad()   # forget this and updates become wrong
loss.backward()
```
- **Snippet explained:** zero_grad clears stale gradients before computing new ones; the same behaviour enables intentional accumulation to simulate large batches.
- **Related terms:** [Backward pass](ml_terms.md#backward-pass), [Training loop](ml_terms.md#training-loop)

---

### Gradient check
*From module: 15_neural_networks*

- **Technical definition:** Compare an analytic gradient with centered finite differences (L(w+e) - L(w-e)) / 2e in float64; pass if the relative error is below about 1e-6.
- **Plain English:** A test that your backward() is right.
- **Example / analogy:** We measured 5.5e-9 for a hand-written 2-layer net.
- **Snippet:**
```python
num = (loss(w + eps) - loss(w - eps)) / (2 * eps)
```
- **Snippet explained:** Perturb one weight each way and divide by the step.
- **Related terms:** [Chain rule (backpropagation through layers)](ml_terms.md#chain-rule-backpropagation-through-layers)

---

### Gradient clipping
*From module: 06_llm_training_and_lora*

- **Technical definition:** Rescaling gradients whose global norm exceeds a threshold before the optimizer step.
- **Plain English:** A speed limit on how hard the model can be corrected at once.
- **Example / analogy:** A governor on an engine.
- **Snippet:**
```python
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```
- **Snippet explained:** Returns the norm before clipping; if it was above 1.0 the gradients are scaled down to 1.0.
- **Related terms:** [Training loop](ml_terms.md#training-loop), [Optimizer (AdamW)](ml_terms.md#optimizer-adamw)

---

### Gradient descent
*From module: 03_ml_foundations*

- **Technical definition:** Iteratively updating parameters opposite to the gradient: p = p - lr * grad.
- **Plain English:** Walk downhill step by step.
- **Example / analogy:** Hiking down a foggy mountain by feeling the slope.
- **Snippet:**
```python
w -= learning_rate * grad_w
```
- **Snippet explained:** Move w a small step against the slope.
- **Related terms:** [Learning rate](ml_terms.md#learning-rate), [Gradient](ml_terms.md#gradient), [Optimizer](ml_terms.md#optimizer)

---

### He initialisation
*From module: 15_neural_networks*

- **Technical definition:** Weights with variance 2/n_in (Kaiming init); ReLU halves the second moment, so Var(w) = 2/n_in keeps Var(y) = Var(x).
- **Plain English:** Xavier corrected for ReLU's half-dead outputs.
- **Example / analogy:** Derivation: Var(y) = n_in Var(w) E[x^2].
- **Snippet:**
```python
torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
```
- **Snippet explained:** Normal weights with std sqrt(2/fan_in).
- **Related terms:** [Xavier (Glorot) initialisation](ml_terms.md#xavier-glorot-initialisation), [ReLU and dead ReLU](ml_terms.md#relu-and-dead-relu), [Vanishing gradient](ml_terms.md#vanishing-gradient)

---

### Hidden layer (MLP)
*From module: 15_neural_networks*

- **Technical definition:** A layer between input and output; an MLP (multi-layer perceptron, feed-forward network) stacks linear maps with nonlinearities, a = f(x W + b).
- **Plain English:** The middle steps where the network builds its own features.
- **Example / analogy:** Pixels to edges to shapes to digits.
- **Snippet:**
```python
h = np.tanh(X @ W1 + b1); out = h @ W2 + b2
```
- **Snippet explained:** Two layers: a tanh hidden layer then a linear output.
- **Related terms:** [Activation function](ml_terms.md#activation-function)

---

### hit@k, MRR, recall@k (retrieval metrics)
*From module: 19_chunking_deep_dive*

- **Technical definition:** hit@k = 1 if any relevant item is in the top k; recall@k = share of all relevant items found in the top k; MRR = mean of 1/rank of the first relevant item.
- **Plain English:** Did the answer show up in the first few results, and how high?
- **Example / analogy:** A quiz where you get 1 point if the right answer is in your top 3 guesses.
- **Snippet:**
```python
rr = 1 / first_rank if first_rank else 0.0
```
- **Snippet explained:** Averaged over all questions; with 74 questions one question is worth 0.0135.
- **Related terms:** [Context precision (chunk attribution)](ml_terms.md#context-precision-chunk-attribution), [Bootstrap confidence interval](ml_terms.md#bootstrap-confidence-interval)

---

### Hyperparameter
*From module: 03_ml_foundations*

- **Technical definition:** A setting chosen before/outside training (learning rate, batch size, epochs, number of layers).
- **Plain English:** A knob that you set, not the model.
- **Example / analogy:** Oven temperature and cooking time.
- **Snippet:**
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
```
- **Snippet explained:** lr is a hyperparameter; weights are parameters.
- **Related terms:** [Learning rate](ml_terms.md#learning-rate), [Batch size](ml_terms.md#batch-size), [Epoch](ml_terms.md#epoch)

---

### Idempotent upsert
*From module: 20_llm_system_lld*

- **Technical definition:** A write that creates or overwrites the record with the same id, so repeating it leaves the same state.
- **Plain English:** Pressing an elevator button twice.
- **Example / analogy:** Running the ingestion job twice gives the same index.
- **Snippet:**
```python
store.upsert([(chunk, embedding)])  # same chunk_id overwrites
```
- **Snippet explained:** No duplicates are created for an existing id.
- **Related terms:** [Content-hash id](ml_terms.md#content-hash-id), [Incremental re-indexing (manifest)](ml_terms.md#incremental-re-indexing-manifest)

---

### Incremental re-indexing (manifest)
*From module: 20_llm_system_lld*

- **Technical definition:** Re-indexing only documents whose content hash differs from a recorded manifest, and removing documents that disappeared.
- **Plain English:** Re-reading only the chapters that changed.
- **Example / analogy:** One edited refund section re-embedded 1 chunk instead of 36.
- **Snippet:**
```python
if manifest.docs.get(doc.doc_id) == doc.content_hash: skip
```
- **Snippet explained:** An unchanged document is skipped without parsing or embedding.
- **Related terms:** [Idempotent upsert](ml_terms.md#idempotent-upsert), [Dead-letter queue](api_and_deployment_terms.md#dead-letter-queue), [Index version](ml_terms.md#index-version)

---

### Index version
*From module: 20_llm_system_lld*

- **Technical definition:** A fingerprint of everything that changes what stored vectors mean (embedder, dimension, chunker settings, schema). A change forces a rebuild.
- **Plain English:** A recipe version printed on every jar.
- **Example / analogy:** Switching to a 256-dim embedder changed the version and rebuilt the index.
- **Snippet:**
```python
content_hash(embedder.name, str(embedder.dim), chunker.name, 'schema1', length=8)
```
- **Snippet explained:** A short hash; compared with the manifest's stored version at the start of each run.
- **Related terms:** [Content-hash id](ml_terms.md#content-hash-id), [Exact cache key and scope](llm_terms.md#exact-cache-key-and-scope)

---

### Inference mode (no_grad)
*From module: 21_llm_inference_engineering*

- **Technical definition:** Running a model without gradient tracking using model.eval() and torch.no_grad().
- **Plain English:** Using the model, not teaching it, so it skips bookkeeping.
- **Example / analogy:** Driving instead of learning to drive: no instructor notes.
- **Snippet:**
```python
model.eval()
with torch.no_grad():
    out = model.generate(**enc)
```
- **Snippet explained:** eval turns off dropout; no_grad saves memory and time.
- **Related terms:** [Training loop](ml_terms.md#training-loop), [KV cache](llm_terms.md#kv-cache)

---

### Inference pipeline
*From module: 10_ml_system_architecture*

- **Technical definition:** The online path that loads a saved artifact once and, per request, validates input, preprocesses, predicts and postprocesses.
- **Plain English:** The 'answering' program, run all the time.
- **Example / analogy:** A student answering exam questions with what they learned.
- **Snippet:**
```python
model, meta = load_artifact()
predict_text(model, 'What a superb hotel')
```
- **Snippet explained:** Loads once, predicts per request, never calls fit.
- **Related terms:** [Training pipeline](ml_terms.md#training-pipeline), [Postprocessing](ml_terms.md#postprocessing)

---

### Iteration (step)
*From module: 05_llm_parameters*

- **Technical definition:** One weight update, i.e. processing one batch (forward, loss, backward, optimizer step).
- **Plain English:** One tweak of the knobs.
- **Example / analogy:** Steps per epoch = ceil(dataset size / batch size).
- **Snippet:**
```python
steps_per_epoch = -(-300 // 32)   # ceiling division = 10
```
- **Snippet explained:** Negative floor-division trick for rounding up.
- **Related terms:** [Batch size](ml_terms.md#batch-size), [Epoch](ml_terms.md#epoch)

---

### KL divergence
*From module: 17_finetuning_alignment_and_benchmarks*

- **Technical definition:** KL(p||q) = sum p log(p/q); in RL fine-tuning, a penalty keeping the policy near the reference.
- **Plain English:** A measure of how far one distribution has drifted from another.
- **Example / analogy:** GRPO often uses the k3 estimator exp(d) - d - 1 with d = log ref - log pi.
- **Snippet:**
```python
kl = torch.exp(d) - d - 1
```
- **Snippet explained:** Non-negative per-sample estimate.
- **Related terms:** [Reference model](llm_terms.md#reference-model), [GRPO (group relative policy optimization)](llm_terms.md#grpo-group-relative-policy-optimization)

---

### L2 regularization
*From module: 15_neural_networks*

- **Technical definition:** Add (wd/2) ||w||^2 to the loss, adding wd w to the gradient, which penalises large weights.
- **Plain English:** A penalty that keeps weights small.
- **Example / analogy:** Equal to weight decay for SGD; different for Adam.
- **Snippet:**
```python
loss = ce + 0.5 * wd * sum((p ** 2).sum() for p in params)
```
- **Snippet explained:** Explicit penalty term.
- **Related terms:** [Decoupled weight decay (AdamW vs L2)](ml_terms.md#decoupled-weight-decay-adamw-vs-l2), [Dropout](ml_terms.md#dropout)

---

### Label
*From module: 03_ml_foundations*

- **Technical definition:** The target value the model should predict (the 'right answer' in supervised learning).
- **Plain English:** The answer key for each training example.
- **Example / analogy:** The house's real sale price.
- **Snippet:**
```python
y = np.array([200, 320])
```
- **Snippet explained:** One label per row of X.
- **Related terms:** [Feature](ml_terms.md#feature), [Supervised learning](ml_terms.md#supervised-learning)

---

### Label masking (-100)
*From module: 04_llm_fundamentals*

- **Technical definition:** Setting label values to -100 so PyTorch's cross-entropy ignores those positions, commonly used for padding (or for prompt tokens in instruction tuning).
- **Plain English:** A 'do not grade this part' marker.
- **Example / analogy:** Without it, padding tokens contribute meaningless loss.
- **Snippet:**
```python
labels = batch['input_ids'].clone()
labels[batch['attention_mask'] == 0] = -100
```
- **Snippet explained:** Copy the ids as labels, then hide padded positions from the loss.
- **Related terms:** [Padding and attention mask](llm_terms.md#padding-and-attention-mask), [Cross-entropy loss](ml_terms.md#cross-entropy-loss), [Fine-tuning](llm_terms.md#fine-tuning)

---

### Label smoothing
*From module: 15_neural_networks*

- **Technical definition:** Replace the one-hot target by (1 - eps) on the true class and eps/(K-1) elsewhere, capping confidence.
- **Plain English:** Never be 100 percent sure.
- **Example / analogy:** Lowers validation loss and improves calibration; accuracy about unchanged.
- **Snippet:**
```python
F.cross_entropy(logits, y, label_smoothing=0.1)
```
- **Snippet explained:** Built-in smoothing option.
- **Related terms:** [Dropout](ml_terms.md#dropout)

---

### Layer normalization
*From module: 15_neural_networks*

- **Technical definition:** Normalise each example over its feature axis (mean and variance per sample), then learned gain and bias; no batch dependence, same train and eval behaviour.
- **Plain English:** Standardise each example by itself.
- **Example / analogy:** Used in the original transformer and GPT-2.
- **Snippet:**
```python
y = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + eps)
```
- **Snippet explained:** Statistics over the last axis.
- **Related terms:** [Batch normalization](ml_terms.md#batch-normalization), [RMSNorm](ml_terms.md#rmsnorm)

---

### Layout analysis (page segmentation)
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** Detecting and classifying page regions (title, text, table, figure, header, footer) and ordering them.
- **Plain English:** Understanding the page before reading it.
- **Example / analogy:** Modern parsers use learned layout models with DocLayNet-style labels.
- **Snippet:**
```python
elements = [dict(kind=k, bbox=b) for k, b in blocks]
```
- **Snippet explained:** A list of typed boxes per page.
- **Related terms:** [Recursive XY-cut](ml_terms.md#recursive-xy-cut), [Reading order](ml_terms.md#reading-order), [Parsed document (layout elements)](llm_terms.md#parsed-document-layout-elements)

---

### Lazy loading
*From module: 09_api_development*

- **Technical definition:** Deferring expensive initialisation (like importing or loading a big model) until it is first needed, then caching it.
- **Plain English:** Don't unpack the heavy thing until someone actually asks for it.
- **Example / analogy:** Not starting the oven until an order for pizza arrives.
- **Snippet:**
```python
_GEN = None
def gen():
    global _GEN
    if _GEN is None:
        _GEN = load_big_model()
    return _GEN
```
- **Snippet explained:** First call pays the loading cost; later calls reuse the cached object.
- **Related terms:** [Model loading](llm_terms.md#model-loading), [Lifespan (startup/shutdown)](api_and_deployment_terms.md#lifespan-startupshutdown)

---

### Learning rate
*From module: 03_ml_foundations*

- **Technical definition:** Hyperparameter controlling the step size of each parameter update.
- **Plain English:** How big each step downhill is.
- **Example / analogy:** Stride length when walking down a hill.
- **Snippet:**
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```
- **Snippet explained:** lr=0.01 scales every update.
- **Related terms:** [Gradient descent](ml_terms.md#gradient-descent), [Hyperparameter](ml_terms.md#hyperparameter)

---

### Learning rate (training)
*From module: 06_llm_training_and_lora*

- **Technical definition:** The scale of each weight update step.
- **Plain English:** How big a step you take after each correction.
- **Example / analogy:** Walking downhill in fog: tiny steps are slow, huge steps overshoot.
- **Snippet:**
```python
AdamW(params, lr=5e-4)
```
- **Snippet explained:** Measured: 5e-7 barely moves, 5e-5 steady, 5e-4 faster on this tiny model.
- **Related terms:** [Optimizer (AdamW)](ml_terms.md#optimizer-adamw), [Training loop](ml_terms.md#training-loop)

---

### Learning-rate schedule (warm-up and cosine decay)
*From module: 15_neural_networks*

- **Technical definition:** lr(t) rises linearly for the warm-up steps, then follows 0.5 (1 + cos(pi progress)) down to a floor.
- **Plain English:** Start gently, finish gently.
- **Example / analogy:** Warm-up avoids divergence while Adam's v is noisy and weights are random.
- **Snippet:**
```python
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
```
- **Snippet explained:** lr_factor returns the multiplier at each step.
- **Related terms:** [Adam bias correction](ml_terms.md#adam-bias-correction), [Momentum (SGD with momentum)](ml_terms.md#momentum-sgd-with-momentum), [Gradient clipping](ml_terms.md#gradient-clipping)

---

### Learning-rate scheduler (warm-up)
*From module: 05_llm_parameters*

- **Technical definition:** A rule changing the learning rate during training, commonly a short warm-up from near zero followed by a gradual decay.
- **Plain English:** Starting a run slowly, then easing off toward the finish.
- **Example / analogy:** Warm-up avoids huge early updates that can destabilise a pretrained model.
- **Snippet:**
```python
sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=10)
sched.step()
```
- **Snippet explained:** Call sched.step() after each optimizer step to update the learning rate.
- **Related terms:** [Learning rate](ml_terms.md#learning-rate), [Optimizer](ml_terms.md#optimizer)

---

### Levenshtein distance
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** Minimum number of substitutions, deletions and insertions turning one sequence into another; O(nm) dynamic programme.
- **Plain English:** The edit count between two strings.
- **Example / analogy:** kitten to sitting is 3.
- **Snippet:**
```python
d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1] + (a[i-1] != b[j-1]))
```
- **Snippet explained:** Classic edit-distance recurrence.
- **Related terms:** [CER and WER](ml_terms.md#cer-and-wer)

---

### Linear separability
*From module: 15_neural_networks*

- **Technical definition:** A dataset is linearly separable if one hyperplane w.x + b = 0 puts all examples of each class on opposite sides.
- **Plain English:** You can split the classes with one straight line (or flat plane).
- **Example / analogy:** AND and OR are separable; XOR is not.
- **Snippet:**
```python
w, b = np.array([1, 1]), -1.5   # separates AND
```
- **Snippet explained:** The line x1 + x2 = 1.5 leaves only (1,1) on the positive side.
- **Related terms:** [Perceptron (artificial neuron)](ml_terms.md#perceptron-artificial-neuron), [XOR problem](ml_terms.md#xor-problem)

---

### Log loss
*From module: 10_ml_system_architecture*

- **Technical definition:** A probabilistic metric penalising confident wrong predictions: -mean(log p(true class)); lower is better.
- **Plain English:** Rewards being confident and right, punishes being confident and wrong.
- **Example / analogy:** A quiz where bold wrong answers lose more points.
- **Snippet:**
```python
log_loss(y_test, model.predict_proba(X_test), labels=list(model.classes_))
```
- **Snippet explained:** Computes cross-entropy from predicted probabilities.
- **Related terms:** [Evaluation metric](ml_terms.md#evaluation-metric)

---

### Log-odds (logit) and odds
*From module: 15_neural_networks*

- **Technical definition:** Odds = p/(1-p); log-odds (logit) = ln(p/(1-p)), the inverse of the sigmoid. In logistic regression the logit is linear in the features.
- **Plain English:** How many times likelier yes is than no, on a log scale.
- **Example / analogy:** p = 0.75 means odds 3:1; a weight of 0.69 doubles the odds per unit of the feature.
- **Snippet:**
```python
odds = np.exp(z)
```
- **Snippet explained:** e^z equals p/(1-p).
- **Related terms:** [Logistic regression (logit model)](ml_terms.md#logistic-regression-logit-model)

---

### Logistic regression (logit model)
*From module: 15_neural_networks*

- **Technical definition:** A binary classifier p = sigmoid(w.x + b) fit by minimising binary cross-entropy; a single neuron with a sigmoid output. The logit w.x + b is the log-odds of class 1.
- **Plain English:** A straight-line classifier that reports a probability.
- **Example / analogy:** Spam probability from word counts: 0.97 means very likely spam; the boundary is where it equals 0.5.
- **Snippet:**
```python
p = 1 / (1 + np.exp(-(X @ w + b)))
```
- **Snippet explained:** Weighted sum plus bias, squashed into (0, 1).
- **Related terms:** [Binary cross-entropy (log loss)](ml_terms.md#binary-cross-entropy-log-loss), [Perceptron (artificial neuron)](ml_terms.md#perceptron-artificial-neuron), [Probability calibration](ml_terms.md#probability-calibration)

---

### Logits
*From module: 03_ml_foundations*

- **Technical definition:** Raw, unnormalised model output scores before softmax.
- **Plain English:** The model's raw scores for each option.
- **Example / analogy:** Judges' scores before converting to percentages.
- **Snippet:**
```python
logits = np.array([2.0, 1.0, 0.1])
```
- **Snippet explained:** Scores for 3 classes.
- **Related terms:** [Softmax](ml_terms.md#softmax), [Cross-entropy loss](ml_terms.md#cross-entropy-loss)

---

### Loss function
*From module: 03_ml_foundations*

- **Technical definition:** A function mapping predictions and targets to a single number that training minimises.
- **Plain English:** A score for how wrong the model is.
- **Example / analogy:** A golf score: lower is better.
- **Snippet:**
```python
loss = np.mean((y_hat - y) ** 2)
```
- **Snippet explained:** Mean squared error.
- **Related terms:** [MSE](ml_terms.md#mse), [Cross-entropy loss](ml_terms.md#cross-entropy-loss), [Gradient descent](ml_terms.md#gradient-descent)

---

### Loss scaling
*From module: 15_neural_networks*

- **Technical definition:** Multiply the loss by a large factor before backward so small fp16 gradients do not underflow, then divide the gradients by it before the update; dynamic scalers adjust it.
- **Plain English:** Turn up the volume before storing quiet numbers.
- **Example / analogy:** 1e-8 becomes 0 in fp16 but survives after a 2^14 scale.
- **Snippet:**
```python
scaler = torch.amp.GradScaler()
```
- **Snippet explained:** Wraps scale, unscale, inf-check and step for fp16 training.
- **Related terms:** [Floating-point formats (fp32, fp16, bf16)](ml_terms.md#floating-point-formats-fp32-fp16-bf16), [Mixed precision (autocast)](ml_terms.md#mixed-precision-autocast), [Gradient clipping](ml_terms.md#gradient-clipping)

---

### Low-rank update
*From module: 06_llm_training_and_lora*

- **Technical definition:** Approximating a weight change ΔW (d_out x d_in) as a product B·A of two thin matrices of rank r.
- **Plain English:** Describing a big change with only a few numbers.
- **Example / analogy:** Describing a face with a few sliders instead of every pixel.
- **Snippet:**
```python
delta = B @ A   # (d_out, r) @ (r, d_in)
```
- **Snippet explained:** The product has full size but only r*(d_out+d_in) free numbers.
- **Related terms:** [LoRA](llm_terms.md#lora), [Rank (r)](llm_terms.md#rank-r)

---

### LSTM (long short-term memory)
*From module: 15_neural_networks*

- **Technical definition:** RNN with a cell state updated additively c_t = f c_{t-1} + i g and gates (input, forget, output) controlling flow; the additive path eases vanishing gradients. GRU merges gates into two.
- **Plain English:** An RNN with a notebook it can add to, erase from and read from.
- **Example / analogy:** Used for sequences before transformers.
- **Snippet:**
```python
c = f * c + i * g; h = o * torch.tanh(c)
```
- **Snippet explained:** Core LSTM update.
- **Related terms:** [RNN (recurrent neural network)](ml_terms.md#rnn-recurrent-neural-network), [BPTT (backpropagation through time)](ml_terms.md#bptt-backpropagation-through-time), [Fixed-size bottleneck](ml_terms.md#fixed-size-bottleneck)

---

### Machine learning
*From module: 03_ml_foundations*

- **Technical definition:** Methods that learn parameters of a function from example data to make predictions on new data.
- **Plain English:** Teaching a program by examples instead of writing rules.
- **Example / analogy:** Learning to spot spam by seeing thousands of emails labelled spam / not spam.
- **Snippet:**
```python
model.fit(X_train, y_train)
model.predict(X_new)
```
- **Snippet explained:** fit learns from examples, predict applies what was learned.
- **Related terms:** [Feature](ml_terms.md#feature), [Label](ml_terms.md#label), [Parameter (model)](ml_terms.md#parameter-model)

---

### Median filter
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** Replaces each pixel by the median of its neighbourhood; removes salt-and-pepper noise while preserving edges.
- **Plain English:** Throws away the odd one out.
- **Example / analogy:** Median then Otsu gave ink F1 0.892 versus 0.770 for a box blur.
- **Snippet:**
```python
scipy.ndimage.median_filter(img, size=3)
```
- **Snippet explained:** Slides a 3x3 window and takes the median.
- **Related terms:** [Morphological operations (erosion, dilation, opening, closing)](ml_terms.md#morphological-operations-erosion-dilation-opening-closing), [Binarisation (thresholding)](ml_terms.md#binarisation-thresholding)

---

### Memory arithmetic (bytes per parameter)
*From module: 17_finetuning_alignment_and_benchmarks*

- **Technical definition:** Training memory per parameter: about 16 bytes for mixed-precision Adam full fine-tuning (weights 2, grads 2, fp32 master 4, two moments 8), before activations.
- **Plain English:** Count what must be stored for every number.
- **Example / analogy:** 7B full fine-tune is about 112 GB; LoRA keeps the base at 2 bytes per parameter.
- **Snippet:**
```python
print(16 * 7e9 / 1e9)
```
- **Snippet explained:** Prints 112.0 (GB).
- **Related terms:** [LoRA](llm_terms.md#lora), [QLoRA](llm_terms.md#qlora)

---

### Memory-bound vs compute-bound
*From module: 21_llm_inference_engineering*

- **Technical definition:** Memory-bound: runtime set by bytes/bandwidth. Compute-bound: set by FLOPs/peak FLOP/s.
- **Plain English:** Waiting for delivery vs waiting for the calculator.
- **Example / analogy:** Decode at batch 1 is memory-bound; prefill is compute-bound.
- **Snippet:**
```python
bound = 'memory' if intensity < ridge else 'compute'
```
- **Snippet explained:** Compare intensity with the ridge.
- **Related terms:** [Arithmetic intensity](ml_terms.md#arithmetic-intensity), [Prefill (prompt processing)](llm_terms.md#prefill-prompt-processing)

---

### Micro-batch
*From module: 21_llm_inference_engineering*

- **Technical definition:** A small slice of a batch fed through a pipeline so that stages can work on different slices at the same time.
- **Plain English:** Chop the batch into smaller pieces so nobody waits.
- **Example / analogy:** Cutting a big cake into slices so every station has something to do.
- **Snippet:**
```python
chunks = batch.chunk(8)
```
- **Snippet explained:** Splits the batch into 8 micro-batches for pipeline scheduling.
- **Related terms:** [Pipeline bubble](ml_terms.md#pipeline-bubble), [Pipeline parallelism (PP)](llm_terms.md#pipeline-parallelism-pp)

---

### Mixed precision
*From module: 06_llm_training_and_lora*

- **Technical definition:** Training with 16-bit math for speed while keeping some state (master weights, optimizer) in fp32.
- **Plain English:** Doing most maths in a smaller number format to go faster.
- **Example / analogy:** Rough sketches for speed, a clean master copy for accuracy.
- **Snippet:**
```python
torch.autocast(device_type='cuda', dtype=torch.bfloat16)
```
- **Snippet explained:** Context manager running eligible ops in bf16.
- **Related terms:** [Quantization (weights)](llm_terms.md#quantization-weights), [GPU memory estimate](llm_terms.md#gpu-memory-estimate)

---

### Mixed precision (autocast)
*From module: 15_neural_networks*

- **Technical definition:** Run most ops in a low-precision format while keeping master weights and sensitive ops (softmax, losses, reductions) in fp32.
- **Plain English:** Cheap numbers for the heavy lifting, careful numbers for the bookkeeping.
- **Example / analogy:** torch.autocast('cpu', dtype=torch.bfloat16).
- **Snippet:**
```python
with torch.autocast('cpu', dtype=torch.bfloat16): out = model(x)
```
- **Snippet explained:** Matmuls run in bf16 inside the block.
- **Related terms:** [Floating-point formats (fp32, fp16, bf16)](ml_terms.md#floating-point-formats-fp32-fp16-bf16), [Loss scaling](ml_terms.md#loss-scaling), [NaN debugging](ml_terms.md#nan-debugging)

---

### ML pipeline
*From module: 10_ml_system_architecture*

- **Technical definition:** An ordered, repeatable sequence of stages (collect, preprocess, feature extraction, train, evaluate, save, serve, monitor) implemented as code.
- **Plain English:** The assembly line that turns raw data into a working model and keeps it working.
- **Example / analogy:** A factory line: each station does one job in order.
- **Snippet:**
```python
from sklearn.pipeline import Pipeline
Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])
```
- **Snippet explained:** Chains feature extraction and a model into one object with fit/predict.
- **Related terms:** [Training pipeline](ml_terms.md#training-pipeline), [Inference pipeline](ml_terms.md#inference-pipeline), [Feature engineering](ml_terms.md#feature-engineering)

---

### Model artifact
*From module: 10_ml_system_architecture*

- **Technical definition:** The saved output of training that serving loads: fitted model/weights plus any preprocessing, usually with metadata.
- **Plain English:** The saved file containing the trained model.
- **Example / analogy:** A printed textbook.
- **Snippet:**
```python
joblib.dump(pipeline, 'artifacts/model.joblib')
```
- **Snippet explained:** Serializes the whole fitted pipeline.
- **Related terms:** [Serialization](python_syntax.md#serialization), [Model registry](api_and_deployment_terms.md#model-registry)

---

### Model monitoring
*From module: 10_ml_system_architecture*

- **Technical definition:** Observing a deployed model's service health (latency, errors) and data/prediction quality (drift, accuracy when labels arrive) with alerts.
- **Plain English:** Watching the model after launch to see if it still works.
- **Example / analogy:** Dashboard in a car.
- **Snippet:**
```python
drift_report('normal traffic', ref_texts, live_texts)
```
- **Snippet explained:** Compares a live feature distribution to the training reference.
- **Related terms:** [Data drift](ml_terms.md#data-drift), [Structured logging](api_and_deployment_terms.md#structured-logging)

---

### Model serialization (joblib)
*From module: 03_ml_foundations*

- **Technical definition:** Saving a trained model object to disk (joblib.dump) and restoring it later (joblib.load).
- **Plain English:** Saving the trained model like a document so you can reopen it.
- **Example / analogy:** Saving a game so you don't replay from the start.
- **Snippet:**
```python
joblib.dump(model, 'model.joblib')
model = joblib.load('model.joblib')
```
- **Snippet explained:** Writes and reads the whole fitted pipeline.
- **Related terms:** [Pipeline (sklearn)](ml_terms.md#pipeline-sklearn)

---

### Momentum (SGD with momentum)
*From module: 15_neural_networks*

- **Technical definition:** v = mu v + g; p -= lr v. Accumulates an exponential average of gradients to smooth noise and speed through consistent directions.
- **Plain English:** A heavy ball rolling downhill.
- **Example / analogy:** mu = 0.9 averages about the last 10 gradients.
- **Snippet:**
```python
v = mu * v + g; p -= lr * v
```
- **Snippet explained:** PyTorch convention of the velocity update.
- **Related terms:** [RMSProp](ml_terms.md#rmsprop), [Adam bias correction](ml_terms.md#adam-bias-correction), [Learning-rate schedule (warm-up and cosine decay)](ml_terms.md#learning-rate-schedule-warm-up-and-cosine-decay)

---

### Morphological operations (erosion, dilation, opening, closing)
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** Binary image operators that shrink (erosion) or grow (dilation) shapes; opening = erode then dilate, closing = dilate then erode.
- **Plain English:** Shaving off specks and filling tiny gaps in strokes.
- **Example / analogy:** Opening removes dust but erodes thin strokes, so ink F1 fell to 0.284 on thin text.
- **Snippet:**
```python
scipy.ndimage.binary_opening(ink, np.ones((3, 3)))
```
- **Snippet explained:** Removes ink blobs smaller than 3x3.
- **Related terms:** [Median filter](ml_terms.md#median-filter), [Connected components](ml_terms.md#connected-components)

---

### MSE
*From module: 03_ml_foundations*

- **Technical definition:** Mean squared error: average of squared differences between prediction and target; used for regression.
- **Plain English:** Average of the squared mistakes.
- **Example / analogy:** Squaring makes big mistakes count extra.
- **Snippet:**
```python
nn.MSELoss()(pred, target)
```
- **Snippet explained:** PyTorch built-in MSE.
- **Related terms:** [Loss function](ml_terms.md#loss-function)

---

### NaN debugging
*From module: 15_neural_networks*

- **Technical definition:** Locate the first non-finite tensor (torch.isfinite), turn on torch.autograd.detect_anomaly, check log(0), sqrt(0), division by zero and exp overflow, lower lr, clip, compare in fp32.
- **Plain English:** Find where the first nan appears, not where you notice it.
- **Example / analogy:** sqrt'(0) is inf; multiplied by 0 it becomes nan.
- **Snippet:**
```python
assert torch.isfinite(loss), 'loss is not finite'
```
- **Snippet explained:** Cheap guard in the training loop.
- **Related terms:** [Gradient clipping](ml_terms.md#gradient-clipping), [Floating-point formats (fp32, fp16, bf16)](ml_terms.md#floating-point-formats-fp32-fp16-bf16), [Loss scaling](ml_terms.md#loss-scaling)

---

### nn.Module
*From module: 03_ml_foundations*

- **Technical definition:** Base class of PyTorch models; subclasses define layers in __init__ and computation in forward.
- **Plain English:** The template every PyTorch model is built from.
- **Example / analogy:** A Lego baseplate for layers.
- **Snippet:**
```python
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)
    def forward(self, x):
        return self.fc(x)
```
- **Snippet explained:** Registers a linear layer and defines how input flows through it.
- **Related terms:** [Tensor](ml_terms.md#tensor), [Training loop](ml_terms.md#training-loop)

---

### Numerical gradient check
*From module: 15_neural_networks*

- **Technical definition:** Comparing an analytic gradient with the central difference (L(theta+h) - L(theta-h)) / 2h; agreement near 1e-10 (h=1e-6, float64) validates a hand-derived backward pass.
- **Plain English:** Measure the slope with a ruler to confirm the algebra.
- **Example / analogy:** Our (p - y) x gradient matched to 4e-11.
- **Snippet:**
```python
(f(w + h*e) - f(w - h*e)) / (2*h)
```
- **Snippet explained:** Perturb one parameter up and down and divide the loss change by 2h.
- **Related terms:** [Binary cross-entropy (log loss)](ml_terms.md#binary-cross-entropy-log-loss)

---

### OCR (optical character recognition)
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** Converting images of text (scans, photos, screenshots) into machine-readable text, usually with positions and confidences.
- **Plain English:** Teaching the computer to read a picture of a page.
- **Example / analogy:** A scanned invoice is OCR'd into text and boxes before any field extraction.
- **Snippet:**
```python
text = ocr.read_region(binarise(gray))
```
- **Snippet explained:** Binarises the page and returns recognised lines.
- **Related terms:** [Binarisation (thresholding)](ml_terms.md#binarisation-thresholding), [Layout analysis (page segmentation)](ml_terms.md#layout-analysis-page-segmentation), [CER and WER](ml_terms.md#cer-and-wer)

---

### Optimizer
*From module: 03_ml_foundations*

- **Technical definition:** Algorithm that updates parameters from gradients (SGD, Adam).
- **Plain English:** The rule that decides how to turn gradients into updates.
- **Example / analogy:** The driver steering down the hill.
- **Snippet:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```
- **Snippet explained:** Adam adapts step sizes per parameter.
- **Related terms:** [Gradient descent](ml_terms.md#gradient-descent), [Learning rate](ml_terms.md#learning-rate)

---

### Optimizer (AdamW)
*From module: 06_llm_training_and_lora*

- **Technical definition:** An algorithm that updates weights from gradients; AdamW uses per-parameter adaptive step sizes plus decoupled weight decay.
- **Plain English:** The coach that decides how big a correction each weight gets.
- **Example / analogy:** A GPS recalculating the route using recent traffic history.
- **Snippet:**
```python
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
```
- **Snippet explained:** Creates an optimizer over all parameters; call opt.step() after backward().
- **Related terms:** [Learning rate (training)](ml_terms.md#learning-rate-training), [Weight decay](ml_terms.md#weight-decay), [Training loop](ml_terms.md#training-loop)

---

### Otsu's method
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** Chooses the global threshold that maximises the between-class variance of the histogram of a gray image.
- **Plain English:** Finds the cut that best separates dark ink from bright paper.
- **Example / analogy:** Great on evenly lit pages, wrong on a shadowed page (F1 0.194 in the module run).
- **Snippet:**
```python
t = int(np.nanargmax((mu_t*omega - mu)**2 / (omega*(1-omega))))
```
- **Snippet explained:** Evaluates every threshold from the cumulative histogram and takes the best.
- **Related terms:** [Binarisation (thresholding)](ml_terms.md#binarisation-thresholding), [Adaptive (local) threshold](ml_terms.md#adaptive-local-threshold)

---

### Overfitting
*From module: 03_ml_foundations*

- **Technical definition:** When a model fits training noise and performs poorly on unseen data; seen as low train loss and high validation loss.
- **Plain English:** Memorising instead of understanding.
- **Example / analogy:** A student who memorises past papers but can't solve new questions.
- **Snippet:**
```python
np.polyfit(x_train, y_train, 12)   # very flexible curve
```
- **Snippet explained:** A degree-12 polynomial on 15 points memorises noise.
- **Related terms:** [Underfitting](ml_terms.md#underfitting), [Train/validation split](ml_terms.md#trainvalidation-split), [Early stopping](ml_terms.md#early-stopping)

---

### Overfitting (training)
*From module: 06_llm_training_and_lora*

- **Technical definition:** The model fits training data (low train loss) but does worse on unseen data (high validation loss).
- **Plain English:** Memorising instead of understanding.
- **Example / analogy:** A student who memorises last year's exam answers.
- **Snippet:**
```python
print(train_loss, val_loss)   # train falls, val rises
```
- **Snippet explained:** Diverging train and validation loss is the warning sign.
- **Related terms:** [Validation set](ml_terms.md#validation-set), [Epoch](ml_terms.md#epoch), [Weight decay](ml_terms.md#weight-decay)

---

### Parameter (model)
*From module: 03_ml_foundations*

- **Technical definition:** A number inside a model that is learned from data (e.g. weights and biases); distinct from hyperparameters set by the human.
- **Plain English:** The knobs the training process turns.
- **Example / analogy:** The tuning dials on a radio.
- **Snippet:**
```python
sum(p.numel() for p in model.parameters())
```
- **Snippet explained:** Counts all learnable numbers; LLMs have billions.
- **Related terms:** [Hyperparameter](ml_terms.md#hyperparameter), [Weight](ml_terms.md#weight), [Gradient descent](ml_terms.md#gradient-descent)

---

### Parameter (weight)
*From module: 04_llm_fundamentals*

- **Technical definition:** A number inside the model that is adjusted during training; collectively they store what the model has learned.
- **Plain English:** The knobs the training process turns.
- **Example / analogy:** distilgpt2 has 81,912,576 of them.
- **Snippet:**
```python
sum(p.numel() for p in model.parameters())
```
- **Snippet explained:** Counts every number in every weight tensor.
- **Related terms:** [Parameter count](llm_terms.md#parameter-count), [Backpropagation](ml_terms.md#backpropagation), [Data type (dtype)](ml_terms.md#data-type-dtype)

---

### pass@k
*From module: 17_finetuning_alignment_and_benchmarks*

- **Technical definition:** Probability that at least one of k samples is correct; estimated unbiasedly from n samples with c correct as 1 - C(n-c,k)/C(n,k).
- **Plain English:** The chance a k-try attempt succeeds.
- **Example / analogy:** n=10, c=3, k=5 gives 0.9167.
- **Snippet:**
```python
1 - np.prod(1 - k/np.arange(n-c+1, n+1))
```
- **Snippet explained:** Stable product form.
- **Related terms:** [Bootstrap confidence interval](ml_terms.md#bootstrap-confidence-interval)

---

### Percentile latency (p50, p95)
*From module: 20_llm_system_lld*

- **Technical definition:** The latency below which p percent of requests fall; nearest-rank takes sorted[ceil(p/100*n)-1].
- **Plain English:** '95 of 100 users waited at most this long.'
- **Example / analogy:** Mean 63.4 ms hid a p95 of 900 ms.
- **Snippet:**
```python
xs[max(0, math.ceil(p / 100 * len(xs)) - 1)]
```
- **Snippet explained:** Sorts samples and picks the rank for the percentile.
- **Related terms:** [Span and trace](api_and_deployment_terms.md#span-and-trace), [Online eval hook](llm_terms.md#online-eval-hook)

---

### Percentiles p50/p95/p99
*From module: 21_llm_inference_engineering*

- **Technical definition:** Values below which 50, 95, 99 percent of samples fall.
- **Plain English:** Tail behaviour that the mean hides.
- **Example / analogy:** p95 ITL 105 ms vs mean 45 ms.
- **Snippet:**
```python
np.percentile(itl, 95)
```
- **Snippet explained:** Linear-interpolated 95th percentile.
- **Related terms:** [ITL (inter-token latency)](llm_terms.md#itl-inter-token-latency), [Goodput (SLO-attaining throughput)](llm_terms.md#goodput-slo-attaining-throughput)

---

### Perceptron (artificial neuron)
*From module: 15_neural_networks*

- **Technical definition:** A unit computing a = f(w.x + b), trained by the perceptron rule w += lr (y - y_hat) x; converges only if the data are linearly separable.
- **Plain English:** One weighted vote followed by a yes/no threshold.
- **Example / analogy:** A spam filter that adds up word scores and flags the email if the sum passes a line.
- **Snippet:**
```python
y_hat = int(x @ w + b > 0)
```
- **Snippet explained:** Dot product plus bias, then a step function giving 0 or 1.
- **Related terms:** [Linear separability](ml_terms.md#linear-separability), [Activation function](ml_terms.md#activation-function), [XOR problem](ml_terms.md#xor-problem)

---

### Pipeline (sklearn)
*From module: 03_ml_foundations*

- **Technical definition:** sklearn object chaining preprocessing steps and a final estimator into one fit/predict unit.
- **Plain English:** Preprocessing and model packaged together.
- **Example / analogy:** An assembly line that always runs the same steps in the same order.
- **Snippet:**
```python
Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])
```
- **Snippet explained:** Two named steps executed in sequence.
- **Related terms:** [Data leakage](ml_terms.md#data-leakage), [Model serialization (joblib)](ml_terms.md#model-serialization-joblib), [TF-IDF](ml_terms.md#tf-idf)

---

### Pipeline bubble
*From module: 21_llm_inference_engineering*

- **Technical definition:** Idle time of pipeline stages during fill and drain, equal to (p-1)/(m+p-1) for p stages and m micro-batches.
- **Plain English:** The dead time at the start and end of the assembly line.
- **Example / analogy:** The first station waits for work only at start; the last station is idle until the first item arrives.
- **Snippet:**
```python
print((4-1)/(16+4-1))  # 0.158
```
- **Snippet explained:** Four stages and 16 micro-batches waste about 16% of the time.
- **Related terms:** [Pipeline parallelism (PP)](llm_terms.md#pipeline-parallelism-pp), [Micro-batch](ml_terms.md#micro-batch)

---

### Postprocessing
*From module: 10_ml_system_architecture*

- **Technical definition:** Turning raw model outputs into the final response: thresholds, label mapping, formatting, abstaining when unsure.
- **Plain English:** Final tidying and decision rules after the model speaks.
- **Example / analogy:** The editor polishing a draft.
- **Snippet:**
```python
label = classes[best] if confidence >= 0.6 else 'uncertain'
```
- **Snippet explained:** Low-confidence predictions become 'uncertain'.
- **Related terms:** [Inference pipeline](ml_terms.md#inference-pipeline), [Confidence](ml_terms.md#confidence)

---

### PPO (proximal policy optimization)
*From module: 17_finetuning_alignment_and_benchmarks*

- **Technical definition:** Policy-gradient algorithm that maximises a clipped surrogate objective min(r*A, clip(r,1-eps,1+eps)*A) with advantages from a value function (GAE).
- **Plain English:** Take improvement steps but refuse steps that change the policy too much.
- **Example / analogy:** r is new/old probability ratio.
- **Snippet:**
```python
torch.min(ratio*adv, torch.clamp(ratio,1-eps,1+eps)*adv)
```
- **Snippet explained:** The clipped objective per token.
- **Related terms:** [GAE (generalized advantage estimation)](ml_terms.md#gae-generalized-advantage-estimation), [KL divergence](ml_terms.md#kl-divergence)

---

### Precision
*From module: 03_ml_foundations*

- **Technical definition:** TP / (TP + FP): of items predicted positive, the fraction truly positive.
- **Plain English:** When I say yes, how often am I right?
- **Example / analogy:** A search engine returning mostly relevant results.
- **Snippet:**
```python
precision_score(y_true, y_pred)
```
- **Snippet explained:** Computes TP/(TP+FP).
- **Related terms:** [Recall](ml_terms.md#recall), [F1 score](ml_terms.md#f1-score), [Confusion matrix](ml_terms.md#confusion-matrix)

---

### Prefix beam search
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** CTC decoding that keeps the best prefixes and sums the probability of all paths reaching each prefix.
- **Plain English:** A smarter decoder than taking the best frame each time.
- **Example / analogy:** It found a with probability 0.64 where greedy gave an empty string.
- **Snippet:**
```python
beams = prefix_beam_search(log_probs, width=5)
```
- **Snippet explained:** Returns the best collapsed strings.
- **Related terms:** [CTC (connectionist temporal classification)](ml_terms.md#ctc-connectionist-temporal-classification)

---

### Preprocessing
*From module: 10_ml_system_architecture*

- **Technical definition:** Cleaning and normalising raw inputs (lowercasing, removing punctuation, handling missing values) so training and serving see consistent data.
- **Plain English:** Tidying the data before the model sees it.
- **Example / analogy:** Washing and chopping vegetables.
- **Snippet:**
```python
re.sub(r'[^a-z0-9\s]', '', text.lower())
```
- **Snippet explained:** Lowercase then delete anything that is not a letter, digit or space.
- **Related terms:** [Feature engineering](ml_terms.md#feature-engineering), [Training/serving skew](ml_terms.md#trainingserving-skew)

---

### Probability calibration
*From module: 15_neural_networks*

- **Technical definition:** The property that predictions of probability p are correct a fraction p of the time; measured with reliability bins, ECE, Brier score or BCE; fixed post hoc by temperature or Platt scaling.
- **Plain English:** Does '80% sure' come true 8 times in 10?
- **Example / analogy:** Scaling logits by 3 keeps accuracy at 0.865 but raises ECE from 0.035 to 0.075.
- **Snippet:**
```python
over = sigmoid(3 * z)
```
- **Snippet explained:** Same decisions, bolder probabilities, worse calibration.
- **Related terms:** [Binary cross-entropy (log loss)](ml_terms.md#binary-cross-entropy-log-loss), [Logistic regression (logit model)](ml_terms.md#logistic-regression-logit-model)

---

### Projection profile
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** Sum of ink pixels along rows or columns; gaps in the profile separate lines (rows) and words or columns (columns).
- **Plain English:** A histogram of ink along one axis.
- **Example / analogy:** Row profile gives text lines; column profile gives words and column gutters.
- **Snippet:**
```python
lines = runs(ink.sum(axis=1) > 0)
```
- **Snippet explained:** Rows with any ink form line runs.
- **Related terms:** [Layout analysis (page segmentation)](ml_terms.md#layout-analysis-page-segmentation), [Recursive XY-cut](ml_terms.md#recursive-xy-cut), [Connected components](ml_terms.md#connected-components)

---

### PSI
*From module: 10_ml_system_architecture*

- **Technical definition:** Population Stability Index: sum over bins of (live% - reference%) x ln(live%/reference%); under 0.1 stable, 0.1-0.25 watch, over 0.25 drift.
- **Plain English:** A single score for how different two distributions are.
- **Example / analogy:** Comparing today's crowd to a normal day's crowd.
- **Snippet:**
```python
np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct))
```
- **Snippet explained:** Per-bin differences weighted by the log ratio.
- **Related terms:** [Data drift](ml_terms.md#data-drift)

---

### Reading order
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** The sequence in which the blocks of a page are read; must be inferred for scans.
- **Plain English:** Which block comes next.
- **Example / analogy:** Swapped columns gave CER 0.79 though every character was present.
- **Snippet:**
```python
els = sorted(els, key=lambda e: (e['page'], e['reading_order']))
```
- **Snippet explained:** Consumers sort by page then reading_order, not array order.
- **Related terms:** [Recursive XY-cut](ml_terms.md#recursive-xy-cut), [Parsed document (layout elements)](llm_terms.md#parsed-document-layout-elements)

---

### Recall
*From module: 03_ml_foundations*

- **Technical definition:** TP / (TP + FN): of all true positives, the fraction found.
- **Plain English:** Of all real yes cases, how many did I catch?
- **Example / analogy:** A smoke alarm that catches every fire.
- **Snippet:**
```python
recall_score(y_true, y_pred)
```
- **Snippet explained:** Computes TP/(TP+FN).
- **Related terms:** [Precision](ml_terms.md#precision), [F1 score](ml_terms.md#f1-score)

---

### Receptive field
*From module: 15_neural_networks*

- **Technical definition:** The input region that can influence one output unit; grows with kernel size and stride across layers.
- **Plain English:** How much of the picture one neuron can see.
- **Example / analogy:** Two stacked 3x3 convs see 5x5.
- **Snippet:**
```python
rf = 1 + sum((k - 1) * jump for k, jump in layers)
```
- **Snippet explained:** Accumulate kernel growth scaled by the cumulative stride.
- **Related terms:** [Convolution (cross-correlation)](ml_terms.md#convolution-cross-correlation)

---

### Recursive XY-cut
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** Top-down page segmentation: cut at the widest empty horizontal band, else the widest empty vertical band, recursively; leaves in recursion order give reading order.
- **Plain English:** Use white space to split a page, top and left first.
- **Example / analogy:** Order WER 0.014 with XY-cut versus 0.554 for a naive y-sort on the same text.
- **Snippet:**
```python
gy = widest_gap(sub.any(axis=1), min_gap_y)
```
- **Snippet explained:** Finds the widest empty row band to cut at.
- **Related terms:** [Reading order](ml_terms.md#reading-order), [Projection profile](ml_terms.md#projection-profile)

---

### Rejection-sampling acceptance rule
*From module: 21_llm_inference_engineering*

- **Technical definition:** Accept draft token x with probability min(1, p(x)/q(x)); on rejection resample from norm(max(0, p - q)); the output is distributed exactly as p.
- **Plain English:** Correcting a biased proposal exactly.
- **Example / analogy:** Empirical frequencies matched the target on a toy distribution.
- **Snippet:**
```python
if rng.random() < min(1, p[x] / q[x]): accept
```
- **Snippet explained:** Accept with probability min(1, p/q).
- **Related terms:** [Speculative decoding](llm_terms.md#speculative-decoding)

---

### ReLU and dead ReLU
*From module: 15_neural_networks*

- **Technical definition:** ReLU(z) = max(0, z), slope 1 for z > 0 and 0 otherwise; a dead ReLU is a unit negative for every input, so its gradient is always zero and it never recovers.
- **Plain English:** A simple on/off ramp that can get stuck off.
- **Example / analogy:** A too-high learning rate pushes a bias very negative and kills the unit; leaky ReLU keeps a small slope.
- **Snippet:**
```python
dead = ((h > 0).sum(axis=0) == 0).sum()
```
- **Snippet explained:** Counts hidden units that never fire on the batch.
- **Related terms:** [Activation function](ml_terms.md#activation-function), [Vanishing gradient](ml_terms.md#vanishing-gradient), [He initialisation](ml_terms.md#he-initialisation)

---

### Reproducibility
*From module: 05_llm_parameters*

- **Technical definition:** Being able to get the same results by rerunning the experiment: fixed seeds, recorded configs, pinned library versions.
- **Plain English:** A recipe precise enough that anyone gets the same cake.
- **Example / analogy:** config.json plus fixed seeds let you compare runs fairly.
- **Snippet:**
```python
torch.manual_seed(0)
```
- **Snippet explained:** Seeding is one ingredient; recording config and versions are the others.
- **Related terms:** [Seed (random seed)](ml_terms.md#seed-random-seed), [Configuration file](python_syntax.md#configuration-file), [Hyperparameter](ml_terms.md#hyperparameter)

---

### Residual connection (skip connection)
*From module: 15_neural_networks*

- **Technical definition:** y = x + f(x): the layer learns a correction; the gradient of y contains the identity, so signal and gradient pass through depth unattenuated. In transformers the running sum is the residual stream.
- **Plain English:** A shortcut road around each layer.
- **Example / analogy:** A 50-layer sigmoid net: gradient ratio 0.0 plain vs 1.27 with residuals.
- **Snippet:**
```python
x = x + block(norm(x))
```
- **Snippet explained:** Pre-norm residual update.
- **Related terms:** [Vanishing gradient](ml_terms.md#vanishing-gradient), [Layer normalization](ml_terms.md#layer-normalization), [RMSNorm](ml_terms.md#rmsnorm)

---

### Ridge point (machine balance)
*From module: 21_llm_inference_engineering*

- **Technical definition:** Intensity where compute and bandwidth roofs meet: peak FLOP/s divided by bandwidth (A100 about 153, H100 about 295 FLOP/byte).
- **Plain English:** The break-even FLOPs per byte.
- **Example / analogy:** Below the ridge you wait for memory.
- **Snippet:**
```python
ridge = 312e12 / 2.039e12
```
- **Snippet explained:** About 153 FLOP per byte for A100.
- **Related terms:** [Roofline model](ml_terms.md#roofline-model), [Memory-bound vs compute-bound](ml_terms.md#memory-bound-vs-compute-bound)

---

### RMSNorm
*From module: 15_neural_networks*

- **Technical definition:** x / sqrt(mean(x^2) + eps) * gain: LayerNorm without mean subtraction and without bias; d parameters instead of 2d.
- **Plain English:** Rescale to a fixed size; skip the centring.
- **Example / analogy:** Used in Llama, Qwen, Gemma.
- **Snippet:**
```python
y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g
```
- **Snippet explained:** Divide by the root mean square of the features.
- **Related terms:** [Layer normalization](ml_terms.md#layer-normalization), [Batch normalization](ml_terms.md#batch-normalization)

---

### RMSProp
*From module: 15_neural_networks*

- **Technical definition:** Divide the step by a running RMS of the gradient per parameter: v = rho v + (1-rho) g^2; p -= lr g / (sqrt(v) + eps).
- **Plain English:** Give every weight its own step size.
- **Example / analogy:** Weights with big gradients take smaller steps.
- **Snippet:**
```python
v = rho * v + (1 - rho) * g * g; p -= lr * g / (v ** 0.5 + eps)
```
- **Snippet explained:** Per-parameter scaling.
- **Related terms:** [Momentum (SGD with momentum)](ml_terms.md#momentum-sgd-with-momentum), [Adam bias correction](ml_terms.md#adam-bias-correction), [Learning-rate schedule (warm-up and cosine decay)](ml_terms.md#learning-rate-schedule-warm-up-and-cosine-decay)

---

### RNN (recurrent neural network)
*From module: 15_neural_networks*

- **Technical definition:** h_t = f(W_xh x_t + W_hh h_{t-1} + b): the same cell is applied at each step, carrying a hidden state.
- **Plain English:** A network with a memory notebook rewritten at every step.
- **Example / analogy:** Reads a sentence word by word.
- **Snippet:**
```python
h = np.tanh(x @ Wxh + h @ Whh + b)
```
- **Snippet explained:** One recurrent update.
- **Related terms:** [BPTT (backpropagation through time)](ml_terms.md#bptt-backpropagation-through-time), [LSTM (long short-term memory)](ml_terms.md#lstm-long-short-term-memory), [Fixed-size bottleneck](ml_terms.md#fixed-size-bottleneck)

---

### Roofline model
*From module: 21_llm_inference_engineering*

- **Technical definition:** Attainable FLOP/s = min(peak FLOP/s, memory bandwidth x arithmetic intensity).
- **Plain English:** Two ceilings; you hit whichever is lower.
- **Example / analogy:** A100: 312 TFLOP/s and 2 TB/s.
- **Snippet:**
```python
attainable = min(peak, bw * intensity)
```
- **Snippet explained:** Compute roof and bandwidth slope in one expression.
- **Related terms:** [Arithmetic intensity](ml_terms.md#arithmetic-intensity), [Ridge point (machine balance)](ml_terms.md#ridge-point-machine-balance)

---

### Row-parallel linear
*From module: 21_llm_inference_engineering*

- **Technical definition:** Splitting W by input rows (and X by columns) so each device produces a partial sum of the full-width output; an all-reduce adds them.
- **Plain English:** Each GPU produces an incomplete full-size answer; add them up.
- **Example / analogy:** Each cook makes a part of every dish; assemble by summing.
- **Snippet:**
```python
Y = np.sum([x @ w for x, w in zip(np.split(X, N, 1), np.split(W, N, 0))], axis=0)
```
- **Snippet explained:** Summing the partial products equals X @ W.
- **Related terms:** [Column-parallel linear](ml_terms.md#column-parallel-linear), [All-reduce](ml_terms.md#all-reduce)

---

### safetensors
*From module: 06_llm_training_and_lora*

- **Technical definition:** A safe, fast tensor file format used by Hugging Face that cannot execute code on load.
- **Plain English:** A weights file format that is quick and safe to open.
- **Example / analogy:** A sealed zip of numbers versus a pickle that could contain surprises.
- **Snippet:**
```python
from safetensors.torch import load_file
w = load_file('model.safetensors')
```
- **Snippet explained:** Loads a dict of tensors from disk.
- **Related terms:** [Checkpoint](ml_terms.md#checkpoint), [state_dict](ml_terms.md#state_dict)

---

### Seed (random seed)
*From module: 05_llm_parameters*

- **Technical definition:** A number initialising a pseudo-random generator so that 'random' results can be repeated.
- **Plain English:** Starting a shuffled deck from the same arrangement every time.
- **Example / analogy:** torch.manual_seed(7) before each generate() call isolates the effect of the parameter you changed.
- **Snippet:**
```python
torch.manual_seed(7)
```
- **Snippet explained:** Fixes PyTorch's global random stream.
- **Related terms:** [Sampling (decoding)](llm_terms.md#sampling-decoding), [Reproducibility](ml_terms.md#reproducibility)

---

### SGD (stochastic gradient descent)
*From module: 05_llm_parameters*

- **Technical definition:** Update rule weight -= lr * gradient computed on a mini-batch; optional momentum smooths updates.
- **Plain English:** The simplest downhill rule: step against the slope.
- **Example / analogy:** Needs careful learning-rate tuning; momentum=0.9 improved the toy loss from 0.39 to 0.24.
- **Snippet:**
```python
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```
- **Snippet explained:** momentum keeps part of the previous step so updates are smoother.
- **Related terms:** [Optimizer](ml_terms.md#optimizer), [Adam and AdamW](ml_terms.md#adam-and-adamw), [Learning rate](ml_terms.md#learning-rate)

---

### Sigmoid saturation
*From module: 15_neural_networks*

- **Technical definition:** For large |z| sigmoid(z) is near 0 or 1 and its slope s(1-s) is near 0, so gradient through that unit vanishes.
- **Plain English:** The curve goes flat at both ends so nothing flows back.
- **Example / analogy:** Max slope 0.25 at z=0.
- **Snippet:**
```python
s = 1 / (1 + np.exp(-z)); slope = s * (1 - s)
```
- **Snippet explained:** Slope formula; peaks at 0.25.
- **Related terms:** [Vanishing gradient](ml_terms.md#vanishing-gradient), [Activation function](ml_terms.md#activation-function), [ReLU and dead ReLU](ml_terms.md#relu-and-dead-relu)

---

### Softmax
*From module: 03_ml_foundations*

- **Technical definition:** Function exp(z_i)/sum(exp(z)) turning logits into probabilities that sum to 1.
- **Plain English:** Scores -> percentages.
- **Example / analogy:** Share of the pie each option gets.
- **Snippet:**
```python
e = np.exp(z - z.max()); probs = e / e.sum()
```
- **Snippet explained:** Subtract max for numerical stability, exponentiate, normalise.
- **Related terms:** [Logits](ml_terms.md#logits), [Cross-entropy loss](ml_terms.md#cross-entropy-loss), [Temperature (softmax)](ml_terms.md#temperature-softmax)

---

### state_dict
*From module: 06_llm_training_and_lora*

- **Technical definition:** A Python dict mapping each layer's parameter name to its tensor.
- **Plain English:** The model's weights as a labelled dictionary.
- **Example / analogy:** A parts list with a label on every part.
- **Snippet:**
```python
sd = model.state_dict()
print(next(iter(sd)))
```
- **Snippet explained:** Prints the first parameter name, e.g. transformer.wte.weight.
- **Related terms:** [Checkpoint](ml_terms.md#checkpoint), [safetensors](ml_terms.md#safetensors)

---

### Straight-through estimator (STE)
*From module: 21_llm_inference_engineering*

- **Technical definition:** Using the identity as the backward pass of a non-differentiable rounding operation.
- **Plain English:** Pretending round has slope 1.
- **Example / analogy:** Used in every QAT implementation.
- **Snippet:**
```python
return w + (wq - w).detach()
```
- **Snippet explained:** Forward value wq, gradient of w.
- **Related terms:** [PTQ vs QAT](llm_terms.md#ptq-vs-qat)

---

### Supervised learning
*From module: 03_ml_foundations*

- **Technical definition:** Learning from input-output pairs (X, y) to predict y for new X.
- **Plain English:** Learning with an answer key.
- **Example / analogy:** Flashcards with the answer on the back.
- **Snippet:**
```python
model.fit(X, y)
```
- **Snippet explained:** Both X and y are given.
- **Related terms:** [Label](ml_terms.md#label), [Machine learning](ml_terms.md#machine-learning)

---

### Table structure recognition
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** Recovering rows, columns and spans of a table, then reading the cells; distinct from table detection.
- **Plain English:** Not just where the table is, but what is in row 2 column 3.
- **Example / analogy:** Ruling lines handle bordered tables; models such as TableFormer handle merged cells.
- **Snippet:**
```python
rows = [[ocr(cell) for cell in row] for row in grid]
```
- **Snippet explained:** One OCR call per cell of the detected grid.
- **Related terms:** [TEDS (tree-edit-distance-based similarity)](ml_terms.md#teds-tree-edit-distance-based-similarity), [Layout analysis (page segmentation)](ml_terms.md#layout-analysis-page-segmentation)

---

### TEDS (tree-edit-distance-based similarity)
*From module: 18_ocr_and_document_parsing*

- **Technical definition:** Table similarity: 1 minus the tree edit distance between the predicted and gold table trees, normalised by the larger tree.
- **Plain English:** A score that notices structure, not just text.
- **Example / analogy:** Merged columns scored text CER 0.000 but TEDS-like 0.645.
- **Snippet:**
```python
teds = 1 - tree_edit_distance(pred, gold) / max(size(pred), size(gold))
```
- **Snippet explained:** Normalised tree edit distance.
- **Related terms:** [Table structure recognition](ml_terms.md#table-structure-recognition), [CER and WER](ml_terms.md#cer-and-wer)

---

### Temperature (softmax)
*From module: 03_ml_foundations*

- **Technical definition:** Divisor applied to logits before softmax: <1 sharpens, >1 flattens the distribution.
- **Plain English:** How adventurous the choice is.
- **Example / analogy:** A low-temperature gas is calm, a high-temperature one is chaotic.
- **Snippet:**
```python
softmax(logits / 0.5)
```
- **Snippet explained:** Dividing by 0.5 doubles the logits, making the top choice more dominant.
- **Related terms:** [Softmax](ml_terms.md#softmax)

---

### Tensor
*From module: 03_ml_foundations*

- **Technical definition:** N-dimensional array (PyTorch) that can track gradients and run on GPUs.
- **Plain English:** A numpy array on steroids.
- **Example / analogy:** A spreadsheet that may have more than two dimensions.
- **Snippet:**
```python
t = torch.tensor([ [1., 2.], [3., 4.] ])
print(t.shape)
```
- **Snippet explained:** A 2x2 tensor; shape is torch.Size([2, 2]).
- **Related terms:** [Autograd](ml_terms.md#autograd), [Broadcasting](ml_terms.md#broadcasting)

---

### TensorDataset
*From module: 06_llm_training_and_lora*

- **Technical definition:** A dataset wrapping one or more tensors, indexed along the first dimension.
- **Plain English:** A table where row i holds the i-th example.
- **Example / analogy:** A spreadsheet with one example per row.
- **Snippet:**
```python
ds = TensorDataset(blocks, labels)
print(len(ds), ds[0][0].shape)
```
- **Snippet explained:** ds[i] returns the i-th row from each tensor as a tuple.
- **Related terms:** [DataLoader](ml_terms.md#dataloader)

---

### TF-IDF
*From module: 03_ml_foundations*

- **Technical definition:** Term frequency x inverse document frequency: weights words that are frequent in a document but rare across documents.
- **Plain English:** Gives rare, informative words more weight than 'the'.
- **Example / analogy:** A rare word like 'refund' says more about a review than 'and'.
- **Snippet:**
```python
TfidfVectorizer(ngram_range=(1, 2)).fit_transform(texts)
```
- **Snippet explained:** Converts texts into weighted word/bigram vectors.
- **Related terms:** [Feature](ml_terms.md#feature), [Pipeline (sklearn)](ml_terms.md#pipeline-sklearn)

---

### Train/test split
*From module: 10_ml_system_architecture*

- **Technical definition:** Separating labelled data into a part used to fit the model and a held-out part used only to estimate performance on unseen data.
- **Plain English:** Keep some examples hidden so you can test honestly.
- **Example / analogy:** Practice problems vs the real exam.
- **Snippet:**
```python
from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
```
- **Snippet explained:** 80/20 split with a fixed seed for reproducibility.
- **Related terms:** [Data leakage](ml_terms.md#data-leakage), [Evaluation metric](ml_terms.md#evaluation-metric)

---

### Train/validation split
*From module: 03_ml_foundations*

- **Technical definition:** Dividing data into training data (used to fit) and held-out validation data (used to evaluate generalisation).
- **Plain English:** Hold some questions back to test the student on unseen material.
- **Example / analogy:** Practice exam vs real exam.
- **Snippet:**
```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
```
- **Snippet explained:** 75/25 split with a fixed seed.
- **Related terms:** [Overfitting](ml_terms.md#overfitting), [Data leakage](ml_terms.md#data-leakage)

---

### Trainable parameters
*From module: 06_llm_training_and_lora*

- **Technical definition:** Parameters with requires_grad=True that the optimizer will update.
- **Plain English:** The weights that are allowed to change.
- **Example / analogy:** The unlocked dials on a mixing desk.
- **Snippet:**
```python
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
```
- **Snippet explained:** Counts elements in all trainable tensors.
- **Related terms:** [LoRA](llm_terms.md#lora), [Optimizer (AdamW)](ml_terms.md#optimizer-adamw)

---

### Training loop
*From module: 03_ml_foundations*

- **Technical definition:** Repeated cycle of forward pass, loss, zero_grad, backward, optimizer step over batches and epochs.
- **Plain English:** The routine every model follows to learn.
- **Example / analogy:** Practice: attempt, get feedback, adjust, repeat.
- **Snippet:**
```python
pred = model(xb)
loss = loss_fn(pred, yb)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```
- **Snippet explained:** The five lines of one training step.
- **Related terms:** [Epoch](ml_terms.md#epoch), [Batch size](ml_terms.md#batch-size), [Optimizer](ml_terms.md#optimizer)

---

### Training pipeline
*From module: 10_ml_system_architecture*

- **Technical definition:** The offline process that prepares data, fits and evaluates a model and writes an artifact plus metadata; runs on a schedule or on demand.
- **Plain English:** The 'teaching' program, run occasionally.
- **Example / analogy:** Writing a textbook.
- **Snippet:**
```python
metadata = train(n_samples=200, C=1.0)
```
- **Snippet explained:** Fits, evaluates and saves model.joblib and metadata.json.
- **Related terms:** [Inference pipeline](ml_terms.md#inference-pipeline), [Model artifact](ml_terms.md#model-artifact)

---

### Training/serving skew
*From module: 10_ml_system_architecture*

- **Technical definition:** A mismatch between how data is processed in training and in production (different cleaning, vocabulary, units), causing silent accuracy loss.
- **Plain English:** The model is taught one way and tested another.
- **Example / analogy:** Learning chess then being asked to play with checkers rules.
- **Snippet:**
```python
wrong = clf.predict(vec.transform(live_texts))   # forgot the lowercase step used in training
```
- **Snippet explained:** Unseen uppercase tokens produce empty vectors and wrong predictions without any error.
- **Related terms:** [Preprocessing](ml_terms.md#preprocessing), [Model monitoring](ml_terms.md#model-monitoring)

---

### Underfitting
*From module: 03_ml_foundations*

- **Technical definition:** Model too simple to capture the pattern; both training and validation error are high.
- **Plain English:** Not learning enough.
- **Example / analogy:** Drawing a straight line through a curve.
- **Snippet:**
```python
np.polyfit(x, y, 1)
```
- **Snippet explained:** A line can't fit a sine wave.
- **Related terms:** [Overfitting](ml_terms.md#overfitting)

---

### Utilisation
*From module: 21_llm_inference_engineering*

- **Technical definition:** The fraction of paid time an accelerator does useful work; effective cost per token is divided by it.
- **Plain English:** How often the GPU is actually busy.
- **Example / analogy:** A shop open ten hours but with customers for two.
- **Snippet:**
```python
eff_cost = cost / 0.2
```
- **Snippet explained:** At 20% busy each token costs 5x as much.
- **Related terms:** [Cost per million tokens](llm_terms.md#cost-per-million-tokens), [Break-even volume](ml_terms.md#break-even-volume)

---

### Validation set
*From module: 05_llm_parameters*

- **Technical definition:** A portion of data held out from training and used to monitor generalisation and tune hyperparameters.
- **Plain English:** Practice exam questions you never trained on.
- **Example / analogy:** The first 300 points trained the model; the next 300 validated it.
- **Snippet:**
```python
X_train, X_val = X[:300], X[300:]
```
- **Snippet explained:** Slicing splits data into two non-overlapping parts.
- **Related terms:** [Overfitting](ml_terms.md#overfitting), [Early stopping](ml_terms.md#early-stopping), [Hyperparameter](ml_terms.md#hyperparameter)

---

### Vanishing gradient
*From module: 15_neural_networks*

- **Technical definition:** Gradients shrink exponentially with depth (or time) because each layer multiplies by a factor below 1, such as sigmoid's slope of at most 0.25; early layers stop learning.
- **Plain English:** The learning signal fades before it reaches the first layers.
- **Example / analogy:** 20 sigmoid layers gave a gradient ratio of about 4e-13.
- **Snippet:**
```python
0.25 ** 20   # about 9e-13
```
- **Snippet explained:** Upper bound on the product of sigmoid slopes over 20 layers.
- **Related terms:** [Exploding gradient](ml_terms.md#exploding-gradient), [Sigmoid saturation](ml_terms.md#sigmoid-saturation)

---

### Vectorizer
*From module: 10_ml_system_architecture*

- **Technical definition:** An object that converts text into numeric vectors (e.g. TfidfVectorizer); it must be fitted on training data and reused unchanged at inference.
- **Plain English:** The translator from words to numbers.
- **Example / analogy:** A dictionary mapping each word to a column number.
- **Snippet:**
```python
vec.transform(['great film'])
```
- **Snippet explained:** Uses the already learned vocabulary; unseen words are ignored.
- **Related terms:** [TF-IDF](ml_terms.md#tf-idf), [fit vs transform](ml_terms.md#fit-vs-transform)

---

### Weight
*From module: 03_ml_foundations*

- **Technical definition:** A learned multiplier applied to an input within a model.
- **Plain English:** How much a feature matters.
- **Example / analogy:** A recipe's amount for each ingredient.
- **Snippet:**
```python
y_hat = w * x + b
```
- **Snippet explained:** w is the weight, b the bias.
- **Related terms:** [Bias (parameter)](ml_terms.md#bias-parameter), [Parameter (model)](ml_terms.md#parameter-model)

---

### Weight decay
*From module: 05_llm_parameters*

- **Technical definition:** A regularisation penalty that gently pulls weights toward zero each update.
- **Plain English:** A small tax on big weights, discouraging over-complicated solutions.
- **Example / analogy:** AdamW(weight_decay=0.01) is a common LLM fine-tuning setting.
- **Snippet:**
```python
torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
```
- **Snippet explained:** Passed as an optimizer argument.
- **Related terms:** [Adam and AdamW](ml_terms.md#adam-and-adamw), [Overfitting](ml_terms.md#overfitting)

---

### Xavier (Glorot) initialisation
*From module: 15_neural_networks*

- **Technical definition:** Weights with variance 2/(n_in + n_out), chosen to keep the variance of activations and gradients roughly constant for tanh/linear layers.
- **Plain English:** Start weights at a size that keeps signals from shrinking or blowing up.
- **Example / analogy:** torch.nn.init.xavier_uniform_.
- **Snippet:**
```python
torch.nn.init.xavier_normal_(layer.weight)
```
- **Snippet explained:** Fills the weights in place with the Glorot variance.
- **Related terms:** [He initialisation](ml_terms.md#he-initialisation), [Vanishing gradient](ml_terms.md#vanishing-gradient), [Layer normalization](ml_terms.md#layer-normalization)

---

### XOR problem
*From module: 15_neural_networks*

- **Technical definition:** XOR (output 1 iff exactly one input is 1) is not linearly separable, so no single neuron can compute it; one hidden layer suffices.
- **Plain English:** The classic proof that depth is needed.
- **Example / analogy:** Best of 20,000 random lines classifies only 3 of the 4 XOR points.
- **Snippet:**
```python
xor = and_(or_(x), nand(x))
```
- **Snippet explained:** A hidden layer of OR and NAND makes the points separable for an AND output neuron.
- **Related terms:** [Linear separability](ml_terms.md#linear-separability), [Activation function](ml_terms.md#activation-function)

---

### ZeRO / FSDP
*From module: 21_llm_inference_engineering*

- **Technical definition:** Training memory optimisation that shards optimizer state (stage 1), gradients (2) and parameters (3) across data-parallel ranks, gathering weights just in time.
- **Plain English:** Instead of every GPU storing everything, each stores a share.
- **Example / analogy:** A library where each branch keeps a share of the books and borrows others when needed.
- **Snippet:**
```python
per_gpu = params * (2/N + 2/N + 12/N)  # ZeRO-3
```
- **Snippet explained:** Memory per GPU divides by N in the ZeRO-3 case.
- **Related terms:** [Data parallelism (DP)](ml_terms.md#data-parallelism-dp), [Tensor parallelism (TP)](llm_terms.md#tensor-parallelism-tp)

---
