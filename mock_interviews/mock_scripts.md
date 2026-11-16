# Mock Interview Scripts (45 minutes each)

Three full scripts. Have a friend read the questions (or read them yourself and answer aloud with a timer), closed notes. Question ids such as `21.M1` refer to [question_bank_15_to_21.md](question_bank_15_to_21.md); drill numbers refer to [live_coding_drills.md](live_coding_drills.md) and [debugging_drills.md](debugging_drills.md). After the session, fill in [scorecard.md](scorecard.md) using the "Scoring hooks" lines.

Rules for the interviewer: ask the question, stay quiet for 20 seconds, then use the follow-ups in order. Do not teach during the session; write the gap down. Scale: 0 blank, 1 needed hints, 2 correct but slow, 3 fluent (same as the scorecard).

---

## Script A: LLM inference engineer

Target: a role that serves open-weight models (GPU or CPU). Heaviest module: 21.

| Min | Block | Content |
|----:|-------|---------|
| 0-5 | Warm-up and project | "Walk me through something you built that serves a model." Listen for: measured numbers, the bottleneck named, one thing that surprised you. |
| 5-12 | Concepts | `21.C3` prefill vs decode, then `21.C4` PagedAttention. |
| 12-22 | Math on the whiteboard | `21.M1` memory for Llama-3-8B, 32 sequences of 4,096 tokens; then `21.M2` decode upper bound on an H100. |
| 22-32 | Live coding (pick one) | Drill 19 (KV bytes) warm-up, then drill 20 (block-hash prefix cache) or drill 18 (int4 pack/unpack). |
| 32-38 | Debugging | `21.D1` junk output in batched generation, then `21.D3` zero prefix-cache hits. If time: debugging drill 23 or 24. |
| 38-45 | Design and wrap-up | `21.A1` 70B chat model, P95 TTFT under 500 ms; close with `21.P3` choosing a batch size. |

Follow-ups, in order:
- After `21.C3`: "Why does quantising weights speed up decode but not prefill?" "What does chunked prefill trade away?" (`21.P2`)
- After `21.M1`: "Now make it fit a 24 GB card. List three changes and the new numbers." (fp8 KV halves 17.2 to 8.6 GB; int4 weights 16 to about 4 GB; fewer sequences or shorter context; GQA is already counted.)
- After `21.M2`: "At what batch does it flip to compute-bound and why do you almost never get there?" (ridge about 150 to 300 FLOP/byte; KV memory fills first.)
- After the coding task: "What happens when two requests share a block and one appends?" (copy-on-write, reference counts.)
- After `21.A1`: "Where would you put speculative decoding, and when does it hurt?" (`21.M3`: low batch, cheap accurate draft; with c = 0.5 and alpha 0.62 it is 0.8x.) "How do you validate your load test?" (open loop, percentiles, goodput against the SLO.)

Strong-answer signals: starts from bytes moved, not from tool names; quotes `2 x layers x kv_heads x head_dim x bytes`; separates TTFT from ITL; says "I would measure" and names how; knows GB versus GiB.
Red flags: "just add more GPUs" without the memory arithmetic; confuses PagedAttention with FlashAttention; cannot say why decode is memory-bound; trusts one benchmark run.

Scoring hooks (scorecard rows to add or update): `LLM inference: memory arithmetic` (21.M1, drill 19), `LLM inference: prefill/decode and batching` (21.C3, 21.P1-P3), `LLM inference: quantisation` (if asked 21.M5, 21.A3), `Live coding: cache/quantiser` (drills 18, 20), `Debugging: inference` (21.D1, 21.D3), `System design: serving` (21.A1).

---

## Script B: RAG and document-intelligence engineer

Target: a role building retrieval over messy documents (PDF, scans, tables). Modules 14, 18, 19, 20.

| Min | Block | Content |
|----:|-------|---------|
| 0-5 | Project | "Describe a RAG system you built. What failed first?" Listen for evaluation numbers, not adjectives. |
| 5-12 | Concepts | `18.C1` how do you decide between extraction and OCR; `19.C1` tokens vs characters and silent truncation. |
| 12-20 | Math | `18.M1` CER and WER on two small examples; `19.M1` windows and overlap storage; `19.M3` is 0.91 vs 0.95 on 74 questions a real difference? |
| 20-32 | Live coding (pick one) | Drill 21 (recursive chunker with offsets) or drill 22 (CTC greedy decode) as a short opener, then drill 23 (token-bucket limiter) if time permits. |
| 32-38 | Debugging | `19.D1` facts at the end of chunks never retrieved; `18.D1` CER 3 percent but users say the output is wrong. |
| 38-45 | Design | `18.A1` or `19.A1` (choose one), then `19.P1` is semantic chunking worth it? |

Follow-ups, in order:
- After `18.C1`: "A scanned PDF returned an empty string and was indexed. What should have caught it?" (a per-page router and a minimum-text check; empty text is a routing signal, not "no content".)
- After `19.C1`: "How do you test that no chunk exceeds the embedder limit?" (count with the embedder's tokenizer; unit test; log truncation.)
- After `19.M3`: "How would you compare them properly?" (paired bootstrap, hit@1 and MRR, context tokens sent, slice by question kind.)
- After the coding task: "Give me the id of a chunk so re-indexing is idempotent." (`19.I2`: hash of doc, chunker version, config, text, occurrence; not position.)
- After `18.A1`: "A VLM parser looks great on the leaderboard. What do you do?" (hard constraints first: privacy, hardware; then test 50 of your own pages; hallucination, omission, no confidence.) "What do you hand to the chunker?" (typed elements with page, bbox in points, reading order, structured tables.)

Strong-answer signals: measures on their own gold set; names failure modes with numbers (silent truncation, merged columns, interleaved columns); keeps page and bbox for citations; versions the index.
Red flags: chooses chunk size by folklore; reports hit@k without context tokens; no plan for scans; treats OCR output as ground truth.

Scoring hooks: `RAG: chunking and retrieval eval` (19.C1, 19.M3, 19.P1), `Documents: OCR and parsing` (18.C1, 18.M1, 18.A1), `Live coding: text processing` (drills 21, 22), `Debugging: retrieval quality` (19.D1, 18.D1), `System design: ingestion` (18.A1 or 19.A1), `LLD: rate limiting and resilience` (drill 23, 20.M2).

---

## Script C: ML engineer fundamentals

Target: a generalist ML/LLM engineer role. Modules 03, 15, 16, 17, a little of 20 and 21.

| Min | Block | Content |
|----:|-------|---------|
| 0-5 | Project | "Explain code you wrote with AI line by line" (question bank item 13): input to processing to output; one limitation. |
| 5-13 | Concepts | `15.C1` nonlinearity and backprop; `16.C1` MHA/GQA/MLA and the KV cache; `17.C1` SFT vs DPO vs GRPO. |
| 13-23 | Math | `15.M1` softmax-CE gradient on three logits; `16.M1` KV cache bytes for Llama-3-8B; `17.M1` DPO loss with beta 0.1. |
| 23-35 | Live coding | Drill 14 (MLP backprop with gradient check) or drill 16 (online softmax) or drill 17 (LoRA layer). One main task, one of the others as a stretch. |
| 35-41 | Debugging | `15.D1` uniform gradient-check error; `17.D1` SFT loss falls but the model does not learn. |
| 41-45 | Design and wrap-up | `15.A1` choose the training recipe for a 1B model; end with `21.C1` pickle vs safetensors as a one-minute closer. |

Follow-ups, in order:
- After `15.C1`: "Why `+=` in backward?" "Why not sigmoid in deep nets?" (slope at most 0.25: about 1e-13 after 20 layers.)
- After `15.M1`: "What changes with a batch of N?" (divide by N; forgetting it gives the same relative error on every parameter.)
- After `16.M1`: "What is MLA caching instead?" (a 512-dim latent plus a 64-dim RoPE key; why RoPE must be decoupled.)
- After `17.M1`: "What is the loss at step 0 and why?" (`ln 2`; policy equals reference.) "What if chosen log-prob falls while accuracy rises?" (`17.D2`.)
- After the coding task: "How do you prove it is right?" (gradient check, reference implementation, tolerance chosen on purpose.)
- After `15.A1`: "Training diverges at step 3k. Walk me through it." (`15.D3`: grad-norm log, bf16 vs fp16, clipping, warm-up, bad batch.)

Strong-answer signals: derives instead of reciting; checks against a reference; mentions the silent failure modes (shift bug, template mismatch, scale errors); numbers with units.
Red flags: cannot write the chain rule for a linear layer; "Adam does not need a learning rate"; cannot say what is in the 16 bytes per parameter.

Scoring hooks: `Neural-network fundamentals` (15.C1, 15.M1, drill 14), `LLM architecture` (16.C1, 16.M1, drill 16), `Fine-tuning and alignment` (17.C1, 17.M1, drill 17), `Debugging: training` (15.D1, 17.D1), `Explaining AI-generated code` (opener), `LLM inference: formats` (21.C1).

---

## After any script

1. Same day: write the three questions you scored 0 or 1 on the scorecard with the exact gap ("could not state the ridge point", not "roofline").
2. Next day: re-answer those aloud, then redo the matching coding drill from scratch.
3. After three scripts, compare the area scores: anything below 2 gets a targeted revisit of that module's `INTERVIEW_GIST.md` before the next mock.
