"""
GOAL:  The low-level design of a production chunker, runnable: a frozen `Chunk` dataclass, a `Chunker` Protocol with a
       strategy registry, config objects with a config hash, content-hash chunk ids, idempotent re-chunking (edit one
       sentence -> only that chunk is re-embedded), versioned indexes behind an alias with an embedding cache,
       streaming of a large document, and a multilingual (CJK) example. Library: chunkerlib/. Tests: test_chunker.py.

RUN (from this step folder):  python main.py        (tests: python -m pytest -q)
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/11_chunker_lld/main.py
EXPECTED OUTPUT (shortened):
    A) registry + one Chunk as JSON          B) ids stable across runs, edit locality (kept / added / removed)
    C) chunker version bump -> new index name, 0 re-embeddings for unchanged text, alias switch
    D) streaming: peak memory batch vs stream   E) multilingual: Chinese text, e5 tokenizer
NEEDS: torch, transformers (cached MiniLM, multilingual-e5-small).  ~20 s.
"""
import sys
import tracemalloc
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from chunkerlib import ChunkConfig, EmbeddingCache, IndexManager, available, chunk_stream, create, diff_chunks  # noqa: E402
from common19 import count_tokens, get_embedder, load_docs  # noqa: E402


def main():
    docs = {d["doc_id"]: d for d in load_docs()}
    hb = docs["employee_handbook"]
    emb = get_embedder()
    cache = EmbeddingCache("all-MiniLM-L6-v2", emb.encode)

    # ---- A) registry, config, one Chunk ------------------------------------------------------------------------------
    print("A) registered chunkers:", available())
    cfg = ChunkConfig(max_tokens=120, overlap_tokens=0)
    md = create("markdown_sections", cfg, count_tokens)
    chunks = md.chunk("employee_handbook", hb["text"])
    print(f"   markdown_sections v{md.version} config_hash={cfg.config_hash} -> {len(chunks)} chunks")
    c = chunks[3]
    print("   chunk[3] as JSON:", c.to_json()[:430], "...")
    try:
        ChunkConfig(max_tokens=64, overlap_tokens=64)
    except ValueError as e:
        print("   invalid config rejected at construction:", e)

    # ---- B) stable ids and edit locality --------------------------------------------------------------------------------
    again = md.chunk("employee_handbook", hb["text"])
    print(f"\nB) chunk it twice: same ids? {[x.chunk_id for x in chunks] == [x.chunk_id for x in again]}")
    mgr = IndexManager("kb", cache)
    name1 = mgr.index_name(md.chunker_id, md.version, cfg.config_hash)
    mgr.build(name1, chunks)
    mgr.switch(name1)
    base_calls = cache.calls
    edited = hb["text"].replace("The budget is 1,200 euros", "The budget is 1,500 euros")
    new = md.chunk("employee_handbook", edited)
    d = mgr.upsert_doc(name1, "employee_handbook", new)
    print(f"   edit one number in the handbook: kept {len(d.kept)}, added {len(d.added)}, removed {len(d.removed)}; "
          f"texts re-embedded: {cache.calls - base_calls} of {len(new)}")
    print(f"   (a position-based id scheme would have invalidated every chunk after the edit)")

    # ---- C) versioned index + alias switch ---------------------------------------------------------------------------------
    md2 = create("markdown_sections", cfg, count_tokens)
    md2.version = "1.1.0"                                          # pretend we fixed a bug in the splitter
    name2 = mgr.index_name(md2.chunker_id, md2.version, cfg.config_hash)
    before = cache.calls
    mgr.build(name2, md2.chunk("employee_handbook", hb["text"]))
    print(f"\nC) chunker 1.0.0 -> 1.1.0: new index {name2!r}")
    print(f"   chunk ids differ (new version in the hash): {not (set(mgr.indexes[name1]) & set(mgr.indexes[name2]))}; "
          f"embeddings recomputed: {cache.calls - before} (cache keyed by text hash)")
    q = emb.encode(["How many weeks of leave does the primary caregiver get?"])[0]
    top_old = mgr.search(q, 1)[0][0]
    mgr.switch(name2)                                              # atomic alias flip; the old index stays for rollback
    top_new = mgr.search(q, 1)[0][0]
    print(f"   alias {mgr.alias!r}; top hit before/after switch: {top_old.section_path[-1]!r} / {top_new.section_path[-1]!r}; "
          f"indexes kept: {len(mgr.indexes)}")

    # ---- D) streaming -------------------------------------------------------------------------------------------------------------
    big = "\n\n".join(d["text"] for d in docs.values()) * 6
    rec = create("recursive", ChunkConfig(max_tokens=150), count_tokens)
    tracemalloc.start()
    batch = rec.chunk("big", big)
    peak_batch = tracemalloc.get_traced_memory()[1]
    tracemalloc.reset_peak()
    n = sum(1 for _ in chunk_stream(iter(big.splitlines(keepends=True)), rec, "big", segment_chars=6000))
    peak_stream = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    print(f"\nD) {len(big):,}-character document: batch -> {len(batch)} chunks, peak {peak_batch / 1e6:.1f} MB; "
          f"stream (6k-char segments) -> {n} chunks, peak {peak_stream / 1e6:.1f} MB")

    # ---- E) multilingual --------------------------------------------------------------------------------------------------------------
    from transformers import AutoTokenizer
    e5 = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    e5_count = lambda t: len(e5(t, add_special_tokens=False)["input_ids"])
    zh = "试用期结束后，员工每周最多可以在家远程工作三天。远程工作需要提前获得经理批准。设备预算为一千二百欧元，每三年可以使用一次。所有收据必须在三十天内上传。"
    cjk = ChunkConfig(max_tokens=18, separators=("\n\n", "\n", "。", "，", " ", ""))
    cs = create("recursive", cjk, e5_count).chunk("zh", zh)
    print(f"\nE) Chinese, e5 tokenizer ({e5_count(zh)} tokens for {len(zh)} chars), separators include '。' and '，':")
    for x in cs:
        print(f"   [{x.token_count:>2} tok] {x.text}")
    with_en = create("recursive", ChunkConfig(max_tokens=18), e5_count).chunk("zh", zh)
    print(f"   with the English-only separators the same text gives {len(with_en)} chunks of "
          f"{[x.token_count for x in with_en]} tokens; first one ends {with_en[0].text[-6:]!r} (cut mid-sentence by the character fallback)")


if __name__ == "__main__":
    main()

# TRY IT: change ChunkConfig(max_tokens=120) to 60 in section A; how many chunks, and does the config_hash change?
# TRY IT: in section B edit a sentence in the FIRST section vs the LAST section; is 'added' still 1?
# BREAK IT: ChunkConfig(max_tokens=64, overlap_tokens=64) raises ValueError at construction time, not deep inside a splitter.
