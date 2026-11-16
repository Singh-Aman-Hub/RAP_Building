# Terms — Semantic Chunking

Folder: `19_chunking_deep_dive/04_semantic_chunking` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [19 README](../README.md)

**Terms in this folder:** [Semantic chunking](#semantic-chunking) · [Cosine distance](#cosine-distance) · [Buffered window](#buffered-window) · [Percentile breakpoint](#percentile-breakpoint) · [Standard-deviation and IQR breakpoints](#standard-deviation-and-iqr-breakpoints) · [Gradient breakpoint](#gradient-breakpoint) · [Min/max size guard](#minmax-size-guard) · [Sparkline](#sparkline) · [ValueError: median](#valueerror-median)

---

## Semantic chunking  `(concept)`
**1. Definition:** Embed sentences, measure cosine distance between neighbours, cut where it spikes.
**2. In plain words:** Cut where the embedding says the topic changes.
**3. Example to quote:** > "It costs a model pass over every sentence."
```python
d = 1.0 - (E[:-1] * E[1:]).sum(axis=1)
```
**Seen in:** main.py, neighbour_distances

## Cosine distance  `(concept)`
**1. Definition:** 1 - cosine similarity. For unit vectors, 1 - dot product. 0 = same direction.
**2. In plain words:** How different two sentences sound.
**3. Example to quote:** > "Embeddings are normalised, so the dot product is the cosine."
**Seen in:** main.py

## Buffered window  `(concept)`
**1. Definition:** Each sentence is embedded together with `buffer` neighbours on each side.
**2. In plain words:** A moving average that stops 'Yes.' from creating a spike.
**3. Example to quote:** > "Buffer 1 triples the tokens embedded."
**Seen in:** main.py, neighbour_distances

## Percentile breakpoint  `(concept)`
**1. Definition:** Cut where distance > np.percentile(d, p). p=85 cuts the top 15% of the distances, in any document.
**2. In plain words:** Always produces cuts, even in a single-topic text.
**3. Example to quote:** > "Percentile breakpoint: Always produces cuts, even in a single-topic text."
**Seen in:** main.py, threshold

## Standard-deviation and IQR breakpoints  `(concept)`
**1. Definition:** threshold = mean + k * std, or mean + k * IQR. With large k a short document gets no cut at all.
**2. In plain words:** Cut only the unusual jumps.
**3. Example to quote:** > "Standard-deviation and IQR breakpoints: Cut only the unusual jumps."
**Seen in:** main.py, threshold

## Gradient breakpoint  `(concept)`
**1. Definition:** Cut where np.gradient(d) is in its top percentile: where the distance changes fastest.
**2. In plain words:** Useful when the whole document is similar.
**3. Example to quote:** > "Gradient breakpoint: Useful when the whole document is similar."
**Seen in:** main.py, breakpoints

## Min/max size guard  `(concept)`
**1. Definition:** After breakpoints: drop the weakest cut next to a chunk below min_tokens; split an oversize chunk at its largest internal distance.
**2. In plain words:** Semantic cuts alone make tiny and huge chunks.
**3. Example to quote:** > "Min/max size guard: Semantic cuts alone make tiny and huge chunks."
**Seen in:** main.py, enforce_sizes

## Sparkline  `(concept)`
**1. Definition:** A one-line text chart using the characters ▁▂▃▄▅▆▇█ scaled between min and max.
**2. In plain words:** Look at the distance curve without matplotlib.
**3. Example to quote:** > "Sparkline: Look at the distance curve without matplotlib."
```python
sparkline(d)
```
**Seen in:** common19.py

## ValueError: median  `(error)`
**1. Definition:** threshold() raises ValueError(method) for an unknown rule.
**2. In plain words:** Only percentile, stdev, iqr are implemented here.
**3. Example to quote:** > "ValueError: median: Only percentile, stdev, iqr are implemented here."
**Seen in:** main.py, BREAK IT
