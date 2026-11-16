# Terms — Hierarchical Parent-Child Retrieval

Folder: `19_chunking_deep_dive/07_hierarchical_parent_child` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [19 README](../README.md)

**Terms in this folder:** [Small-to-big (parent-document retrieval)](#small-to-big-parent-document-retrieval) · [parent_id and child_ids](#parent_id-and-child_ids) · [Docstore](#docstore) · [Sentence-window retrieval](#sentence-window-retrieval) · [Auto-merging retrieval](#auto-merging-retrieval) · [RAPTOR](#raptor) · [Extractive summariser (mock)](#extractive-summariser-mock) · [KMeans clustering](#kmeans-clustering) · [Post-processor](#post-processor)

---

## Small-to-big (parent-document retrieval)  `(concept)`
**1. Definition:** Match small child chunks, return their parent (section) to the LLM.
**2. In plain words:** Find with a small net, hand over the page.
**3. Example to quote:** > "Only leaves are embedded; parents are fetched by parent_id."
**Seen in:** main.py, small_to_big

## parent_id and child_ids  `(concept)`
**1. Definition:** Each node stores one link up and the list of children.
**2. In plain words:** The tree is a dictionary of nodes.
**3. Example to quote:** > "parent_id and child_ids: The tree is a dictionary of nodes."
**Seen in:** main.py, Node

## Docstore  `(concept)`
**1. Definition:** A key-value store {node_id: node} for the parents (and leaves) text; the vector DB holds only leaf vectors and payload with parent_id.
**2. In plain words:** Where the big text lives.
**3. Example to quote:** > "Docstore: Where the big text lives."
**Seen in:** main.py, section A

## Sentence-window retrieval  `(concept)`
**1. Definition:** Index single sentences; send the hit plus w neighbours each side.
**2. In plain words:** Positional neighbours instead of a tree.
**3. Example to quote:** > "Sentence-window retrieval: Positional neighbours instead of a tree."
**Seen in:** main.py, sentence_window

## Auto-merging retrieval  `(concept)`
**1. Definition:** If more than `ratio` of a parent's children are retrieved, replace them by the parent; repeat upward.
**2. In plain words:** Enough pieces of the puzzle: take the whole piece.
**3. Example to quote:** > "Auto-merging retrieval: Enough pieces of the puzzle: take the whole piece."
**Seen in:** main.py, auto_merge

## RAPTOR  `(concept)`
**1. Definition:** Cluster chunks, summarise each cluster, repeat to a root; index every level (arXiv 2401.18059).
**2. In plain words:** Summaries of summaries.
**3. Example to quote:** > "RAPTOR: Summaries of summaries."
**Seen in:** main.py, build_raptor

## Extractive summariser (mock)  `(concept)`
**1. Definition:** Picks the 2 members closest to the cluster centroid instead of writing a new text, so source spans stay honest.
**2. In plain words:** Stand-in for an LLM summariser.
**3. Example to quote:** > "Extractive summariser: Stand-in for an LLM summariser."
**Seen in:** main.py, extractive_summary

## KMeans clustering  `(library)`
**1. Definition:** sklearn.cluster.KMeans(k, n_init, random_state).fit_predict(X) assigns each row to one of k clusters.
**2. In plain words:** Group similar chunks.
**3. Example to quote:** > "KMeans clustering: Group similar chunks."
**Seen in:** main.py, build_raptor

## Post-processor  `(concept)`
**1. Definition:** A function turning ranked leaf ids into the contexts actually sent (parents, merged nodes, windows).
**2. In plain words:** What happens between retrieval and prompt.
**3. Example to quote:** > "Post-processor: What happens between retrieval and prompt."
**Seen in:** common19.evaluate(postprocess=...)
