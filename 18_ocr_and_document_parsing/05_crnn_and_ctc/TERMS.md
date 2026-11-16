# Terms — CRNN and CTC

Folder: `18_ocr_and_document_parsing/05_crnn_and_ctc` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [18 README](../README.md)

**Terms in this folder:** [CRNN (convolutional recurrent neural network)](#crnn-convolutional-recurrent-neural-network) · [CTC (Connectionist Temporal Classification)](#ctc-connectionist-temporal-classification) · [Blank token](#blank-token) · [Collapsing (B function)](#collapsing-b-function) · [Path, alignment](#path-alignment) · [Greedy (best-path) decoding](#greedy-best-path-decoding) · [Prefix beam search](#prefix-beam-search) · [Blank-only plateau](#blank-only-plateau) · [CTC input/target length constraint](#ctc-inputtarget-length-constraint) · [nn.CTCLoss tensor layout](#nnctcloss-tensor-layout) · [Bidirectional GRU/LSTM](#bidirectional-grulstm) · [Character error rate (CER)](#character-error-rate-cer) · [TrOCR, PARSeq](#trocr-parseq) · [RuntimeError: input_lengths must be of size batch_size](#runtimeerror-input_lengths-must-be-of-size-batch_size)

---

## CRNN (convolutional recurrent neural network)  `(concept)`
**1. Definition:** A text-line recogniser: CNN feature extractor, bidirectional recurrent layer over the width axis, linear classifier per column, trained with CTC (Shi et al., 2015).
**2. In plain words:** Image width becomes time, so a sequence model can "read" left to right. 133,317 parameters here.
**3. Example to quote:** > "A CRNN needs no character boxes: only the image and its text."
**Seen in:** `main.py`, `CRNN`

## CTC (Connectionist Temporal Classification)  `(concept)`
**1. Definition:** A loss that equals -log of the sum of probabilities of all frame-level paths that collapse to the target text.
**2. In plain words:** "I do not know where each letter is; add up every way the frames could spell this word."
**3. Example to quote:** > "With T=4 and the label ab there are 15 valid paths; the brute-force loss and nn.CTCLoss both gave 1.2996."
**Seen in:** `main.py`, section 2

## Blank token  `(concept)`
**1. Definition:** A special extra class (index 0) meaning "no character at this frame", used to separate repeated letters and fill gaps.
**2. In plain words:** Not a space. The silence between letters.
**3. Example to quote:** > "Without a blank the network could not write aa; with it a-a collapses to aa while aaa collapses to a."
**Seen in:** `main.py`, `collapse`

## Collapsing (B function)  `(concept)`
**1. Definition:** Mapping a path to text by merging consecutive repeats and then deleting blanks.
**2. In plain words:** aa-bb becomes ab.
**3. Example to quote:** > "Merge repeats first, then remove blanks; the order matters."
```python
if p != prev and p != BLANK: out.append(p)
```
**Seen in:** `main.py`, section 1

## Path, alignment  `(concept)`
**1. Definition:** A sequence of one class per time step; many paths map to the same text.
**2. In plain words:** One of the many ways the 24 frames could spell the word.
**3. Example to quote:** > "CTC sums over paths, not over texts."
**Seen in:** `main.py`, section 2

## Greedy (best-path) decoding  `(concept)`
**1. Definition:** Take the argmax class at every step and collapse.
**2. In plain words:** Fast but myopic: finds the best path, not the best text.
**3. Example to quote:** > "In the toy case greedy outputs the empty string (0.36) while P("a") = 0.64."
**Seen in:** `main.py`, `greedy_decode`

## Prefix beam search  `(concept)`
**1. Definition:** Beam search over collapsed prefixes that tracks p_blank and p_non-blank for each prefix, so all paths with the same prefix are summed.
**2. In plain words:** Keeps several candidate spellings and handles repeated letters correctly.
**3. Example to quote:** > "Beam 5 improved exact match from 0.673 to 0.687 and CER from 0.122 to 0.117; a language model can be added on top."
**Seen in:** `main.py`, `prefix_beam_search`

## Blank-only plateau  `(concept)`
**1. Definition:** An early phase of CTC training where the model outputs only blanks and the loss sits flat (about 4 here) before letters are discovered.
**2. In plain words:** The safe guess first, the letters later: loss 4.1 for 200 steps, below 1 by step 600.
**3. Example to quote:** > "If the plateau does not break, try more steps, BatchNorm, a lower learning rate or a shorter alphabet."
**Seen in:** Execution trace

## CTC input/target length constraint  `(concept)`
**1. Definition:** Valid alignments need T >= len(label) + (number of adjacent repeated letters); otherwise the loss is infinite.
**2. In plain words:** The image must have enough columns to spell the text.
**3. Example to quote:** > "`zero_infinity=True` zeroes impossible samples instead of poisoning training."
**Seen in:** `main.py`, `nn.CTCLoss`

## nn.CTCLoss tensor layout  `(syntax)`
**1. Definition:** log_probs `(T, N, C)` after log_softmax, targets as a 1-D concatenation or `(N, S)`, input_lengths `(N,)`, target_lengths `(N,)`.
**2. In plain words:** Time first, not batch first.
**3. Example to quote:** > "Forgetting `permute(1, 0, 2)` raises "input_lengths must be of size batch_size"."
```python
log_probs = model(x).log_softmax(2).permute(1, 0, 2)
```
**Seen in:** `main.py`, training loop

## Bidirectional GRU/LSTM  `(concept)`
**1. Definition:** Two recurrent layers, one reading left to right and one right to left, whose outputs are concatenated per step.
**2. In plain words:** Each column sees what is on both sides, which resolves ambiguous letters.
**3. Example to quote:** > "GRU(256, 64, bidirectional=True) outputs 128 features per column."
**Seen in:** `main.py`, `CRNN`

## Character error rate (CER)  `(concept)`
**1. Definition:** Edit distance between reference and hypothesis divided by the number of reference characters.
**2. In plain words:** How many single-letter fixes per letter. Here 0.122 (greedy).
**3. Example to quote:** > "CER is defined in step 06; here it is a quick edit-distance function."
**Seen in:** `main.py`, `edit_distance`

## TrOCR, PARSeq  `(concept)`
**1. Definition:** Attention-based text-line recognisers: TrOCR pairs a ViT encoder with a text transformer decoder; PARSeq uses permuted autoregressive decoding with iterative refinement.
**2. In plain words:** Newer recognisers that generate characters one by one instead of aligning frames.
**3. Example to quote:** > "They are usually more accurate on hard text but slower and can hallucinate; CTC models remain the speed choice."
**Seen in:** `WALKTHROUGH.md`, theory

## RuntimeError: input_lengths must be of size batch_size  `(error)`
**1. Definition:** CTCLoss got log_probs with the wrong axis order, so it read the time axis as the batch.
**2. In plain words:** Shape is (B, T, C) instead of (T, B, C).
**3. Example to quote:** > "I assert log_probs.shape == (T, B, C) before the loss."
**Seen in:** BREAK IT in `WALKTHROUGH.md`
