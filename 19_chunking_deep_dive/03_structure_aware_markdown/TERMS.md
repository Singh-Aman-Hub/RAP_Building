# Terms — Structure-Aware Markdown Chunking

Folder: `19_chunking_deep_dive/03_structure_aware_markdown` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [19 README](../README.md)

**Terms in this folder:** [Block (heading, paragraph, list, table, code)](#block-heading-paragraph-list-table-code) · [Fenced code block](#fenced-code-block) · [Heading tree](#heading-tree) · [Breadcrumb header](#breadcrumb-header) · [Atomic block](#atomic-block) · [Table row groups with repeated header](#table-row-groups-with-repeated-header) · [Tiny-section merging](#tiny-section-merging) · [Dataclass](#dataclass)

---

## Block (heading, paragraph, list, table, code)  `(concept)`
**1. Definition:** The first-pass parse of Markdown into typed spans with character offsets.
**2. In plain words:** Know what each piece of the document IS before cutting.
**3. Example to quote:** > "Parse blocks first; never cut inside a table or a fence."
**Seen in:** main.py, parse_blocks

## Fenced code block  `(concept)`
**1. Definition:** Lines between two ``` fences; a '# comment' inside is not a heading.
**2. In plain words:** One atomic block.
**3. Example to quote:** > "Without fence handling '# Report a suspected incident' becomes a fake heading."
**Seen in:** main.py, parse_blocks

## Heading tree  `(concept)`
**1. Definition:** Sections nested by heading level, built with a stack: a heading pops all open sections of the same or deeper level.
**2. In plain words:** The outline of the document.
**3. Example to quote:** > "The stack makes each section know its parent."
```python
while stack[-1].level >= b.level: stack.pop()
```
**Seen in:** main.py, build_tree

## Breadcrumb header  `(concept)`
**1. Definition:** 'Doc > Section > Subsection' prepended to embed_text and context_text.
**2. In plain words:** The address of the chunk.
**3. Example to quote:** > "It lifted dense hit@1 from 0.73 to 0.78 at 200 tokens."
**Seen in:** main.py, split_document

## Atomic block  `(concept)`
**1. Definition:** A block never split by the section packer: only split inside when it is larger than the budget, and then in its own way.
**2. In plain words:** Tables and code stay in one piece if they fit.
**3. Example to quote:** > "Atomic means indivisible unless it cannot fit."
**Seen in:** main.py, section_units

## Table row groups with repeated header  `(concept)`
**1. Definition:** An oversize table is cut into groups of rows; the two header lines are put in front of every group.
**2. In plain words:** Every piece still knows its columns.
**3. Example to quote:** > "A row without its header row is meaningless."
**Seen in:** main.py, split_block

## Tiny-section merging  `(concept)`
**1. Definition:** Adjacent sibling sections below min_tokens are merged (heading kept inside the text) if the result fits max_tokens.
**2. In plain words:** Stops one-line chunks from polluting the index.
**3. Example to quote:** > "Merge tiny siblings, never tiny strangers."
**Seen in:** main.py, merge_tiny

## Dataclass  `(syntax)`
**1. Definition:** @dataclass generates __init__, __repr__ and __eq__ from annotated fields; field(default_factory=list) gives each instance its own list.
**2. In plain words:** A struct with free methods.
**3. Example to quote:** > "Block and Section are dataclasses."
```python
@dataclass
class Block:
    kind: str
```
**Seen in:** main.py, Block / Section
