# Terms — Recursive Splitter Rebuilt

Folder: `19_chunking_deep_dive/02_recursive_splitter_rebuilt` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [19 README](../README.md)

**Terms in this folder:** [Recursive character splitting](#recursive-character-splitting) · [Separator hierarchy](#separator-hierarchy) · [length_function](#length_function) · [keep_separator](#keep_separator) · [Merge with overlap](#merge-with-overlap) · [Regex capture group in re.split](#regex-capture-group-in-resplit) · [Language-specific separators](#language-specific-separators) · [Offsets by str.find](#offsets-by-strfind)

---

## Recursive character splitting  `(concept)`
**1. Definition:** Try separators from coarse to fine; split on the first one that occurs; merge small pieces up to chunk_size; recurse with finer separators only into pieces that are still too big.
**2. In plain words:** Cut at the biggest natural break that works.
**3. Example to quote:** > "Recursive splitting respects paragraphs first, words last."
```python
chunks += self.split_text(p, rest)
```
**Seen in:** main.py, RecursiveSplitter.split_text

## Separator hierarchy  `(concept)`
**1. Definition:** An ordered list such as ['\n\n', '\n', ' ', ''] from the strongest to the weakest boundary. '' means 'cut between characters'.
**2. In plain words:** A ladder of cut points.
**3. Example to quote:** > "The last separator is the emergency brake."
```python
GENERIC = ['\n\n', '\n', ' ', '']
```
**Seen in:** main.py, GENERIC / MARKDOWN / PYTHON

## length_function  `(syntax)`
**1. Definition:** The callable measuring a piece. Default len() (characters). Pass count_tokens for token budgets.
**2. In plain words:** The ruler.
**3. Example to quote:** > "chunk_size is meaningless without a length_function."
```python
RecursiveSplitter(128, 0, length_function=count_tokens)
```
**Seen in:** main.py, __init__

## keep_separator  `(syntax)`
**1. Definition:** False drops the separator, 'start' keeps it at the start of the next piece, 'end' at the end of the previous one.
**2. In plain words:** Does the heading stay with its text?
**3. Example to quote:** > "keep='start' keeps '## Heading' attached to its section; 'end' glues it to the previous one."
**Seen in:** main.py, split_keep

## Merge with overlap  `(concept)`
**1. Definition:** Pack pieces into a chunk until adding one exceeds chunk_size; close the chunk; drop pieces from its FRONT until at most chunk_overlap remains; continue.
**2. In plain words:** The next chunk starts with the tail of the previous one.
**3. Example to quote:** > "Overlap is made of whole pieces, so it is at most chunk_overlap tokens."
**Seen in:** main.py, merge

## Regex capture group in re.split  `(syntax)`
**1. Definition:** re.split with a capturing group returns the separators too: [t0, sep1, t1, ...].
**2. In plain words:** Lets you glue each separator to a neighbour.
**3. Example to quote:** > "re.split(f'({sep})', text) keeps the separators in the list."
```python
re.split('(\n#{1,6} )', text)
```
**Seen in:** main.py, split_keep

## Language-specific separators  `(concept)`
**1. Definition:** Separator lists for a syntax: Markdown headings and fences, Python class/def. Space-indented methods are not matched by '\n\tdef '.
**2. In plain words:** Tell the splitter where code or Markdown naturally breaks.
**3. Example to quote:** > "Check that the separator regex matches your indentation style."
**Seen in:** main.py, PYTHON

## Offsets by str.find  `(syntax)`
**1. Definition:** After splitting, the position of each chunk in the original is recovered with text.find(chunk, start).
**2. In plain words:** How we know where a chunk came from.
**3. Example to quote:** > "LangChain's add_start_index does the same search."
**Seen in:** main.py, split_document
