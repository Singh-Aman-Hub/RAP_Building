# Terms — Qdrant Fundamentals

Folder: `14_rag_and_retrieval/11_qdrant_fundamentals` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [14 README](../README.md)

**Terms in this folder:** [Qdrant](#qdrant) · [Collection](#collection) · [Point](#point) · [Payload](#payload) · [Point id (uuid5)](#point-id-uuid5) · [Upsert](#upsert) · [Idempotent ingestion](#idempotent-ingestion) · [Distance metric (Cosine, Dot, Euclid)](#distance-metric-cosine-dot-euclid) · [VectorParams](#vectorparams) · [HNSW (index and config)](#hnsw-index-and-config) · [Payload index](#payload-index) · [PayloadSchemaType](#payloadschematype) · [Filter (must, must_not, should)](#filter-must-must_not-should) · [FieldCondition, MatchValue, MatchAny, Range](#fieldcondition-matchvalue-matchany-range) · [query_points](#query_points) · [score_threshold](#score_threshold) · [with_payload and with_vectors](#with_payload-and-with_vectors) · [retrieve (get by id)](#retrieve-get-by-id) · [scroll (pagination)](#scroll-pagination) · [Delete by filter (FilterSelector)](#delete-by-filter-filterselector) · [count](#count) · [Client modes (:memory:, path, url)](#client-modes-memory-path-url) · [Local mode vs server mode](#local-mode-vs-server-mode) · [Named vectors](#named-vectors) · [Quantization (in Qdrant)](#quantization-in-qdrant) · [qdrant_client.models](#qdrant_clientmodels) · [Collection lifecycle calls](#collection-lifecycle-calls) · [tempfile.TemporaryDirectory](#tempfiletemporarydirectory)

---

## Qdrant  `(concept)`
**1. Definition:** An open-source vector database written in Rust that stores points (id + vector + payload) in collections and answers approximate nearest-neighbour queries with metadata filters.
**2. In plain words:** The database that does what step 04 built by hand: it holds the chunk vectors and returns the closest ones, with filters.
**3. Example to quote:** > "Qdrant is a purpose-built vector DB: HNSW search, payload filtering and persistence behind one API."
**Seen in:** `main.py`, whole file

## Collection  `(concept)`
**1. Definition:** One named vector space in Qdrant; every point in it has the same vector size and distance metric, both fixed at creation.
**2. In plain words:** Like a table with a vector column. Here: `kb`, size 384, cosine.
**3. Example to quote:** > "A collection's size and distance can't change later, so I read the size from the model, not from memory."
**Seen in:** `main.py`, part B (`create_collection`)

## Point  `(concept)`
**1. Definition:** The unit of storage in Qdrant: an id, a vector (or several named vectors) and an optional payload.
**2. In plain words:** What other tools call a document or row. One point per chunk (36 here).
**3. Example to quote:** > "A point is id plus vector plus payload; I store the chunk text in the payload so a hit is self-contained."
**Seen in:** `main.py`, part C (`models.PointStruct`)

## Payload  `(concept)`
**1. Definition:** The JSON object stored next to a point's vector; its fields can be returned with results and used in filters.
**2. In plain words:** The chunk's metadata: `chunk_id`, `doc_id`, `section`, `text`, `n_words`.
**3. Example to quote:** > "Payload is where metadata lives; filters run on it, and with_payload returns it."
**Seen in:** `main.py`, parts C and E

## Point id (uuid5)  `(concept)`
**1. Definition:** Qdrant ids must be unsigned 64-bit integers or UUIDs. `uuid.uuid5(namespace, name)` derives a UUID deterministically from a string.
**2. In plain words:** `rule_0` is rejected; `uuid5(NAMESPACE, 'rule_0')` is accepted and always the same, so re-ingestion replaces instead of duplicating.
**3. Example to quote:** > "I derive ids with uuid5 from the chunk id, which makes ingestion idempotent."
**Seen in:** `main.py`, part C (`point_id`)

## Upsert  `(function)`
**1. Definition:** `client.upsert(collection_name, points, wait)` inserts points or replaces those with an existing id. `wait=True` returns after the write is applied and searchable.
**2. In plain words:** Upserting the same 36 points twice still gives 36 points.
**3. Example to quote:** > "Upsert plus deterministic ids means I can re-run the pipeline safely."
**Seen in:** `main.py`, part C

## Idempotent ingestion  `(concept)`
**1. Definition:** Running the same load twice leaves the system in the same state as running it once.
**2. In plain words:** uuid5 ids plus upsert give 36 points both times.
**3. Example to quote:** > "I make ingestion idempotent so a retry never duplicates chunks."
**Seen in:** `main.py`, part C

## Distance metric (Cosine, Dot, Euclid)  `(concept)`
**1. Definition:** The similarity function of a collection: `models.Distance.COSINE`, `DOT` or `EUCLID`. For Cosine, Qdrant normalises vectors and the score is cosine similarity, higher is closer.
**2. In plain words:** We use Cosine and also normalise embeddings ourselves, so cosine and dot would rank identically.
**3. Example to quote:** > "Match the metric to how the embedding model was trained; for normalised vectors cosine and dot agree."
**Seen in:** `main.py`, parts B and E

## VectorParams  `(class)`
**1. Definition:** `models.VectorParams(size, distance)`: the configuration of one vector field in a collection. Optional fields include `hnsw_config`, `on_disk` and `quantization_config`.
**2. In plain words:** Our call sets only size=384 and distance=COSINE.
**3. Example to quote:** > "VectorParams is the schema of the vector column."
**Seen in:** `main.py`, part B

## HNSW (index and config)  `(concept)`
**1. Definition:** Hierarchical Navigable Small World: a layered proximity graph that Qdrant builds for approximate nearest-neighbour search. Tunable via `hnsw_config` (`m`, `ef_construct`) and a per-query `ef`.
**2. In plain words:** Not tuned in this step; defaults are fine for 36 points. Larger `m` / `ef` means better recall but more memory and time.
**3. Example to quote:** > "HNSW trades a little recall for a big speed-up; m and ef are the recall/latency knobs."
**Seen in:** `WALKTHROUGH.md`, names table (not set in `main.py`)

## Payload index  `(concept)`
**1. Definition:** An index on one payload field (`create_payload_index(collection, field_name, field_schema)`) so filters on that field do not scan every point.
**2. In plain words:** We index `doc_id` as KEYWORD and `n_words` as INTEGER. Local mode accepts the call and ignores it.
**3. Example to quote:** > "I create a payload index per filtered field before bulk loading; the vector index does not help filters."
**Seen in:** `main.py`, part D

## PayloadSchemaType  `(class)`
**1. Definition:** The enum of payload index types: `KEYWORD` (exact strings), `INTEGER`, `FLOAT`, `BOOL`, `GEO`, `DATETIME`, `TEXT`, `UUID`.
**2. In plain words:** KEYWORD for doc_id because we match it exactly; INTEGER for n_words because we range over it.
**3. Example to quote:** > "Pick the index type that matches the filter you run."
**Seen in:** `main.py`, part D

## Filter (must, must_not, should)  `(class)`
**1. Definition:** `models.Filter` combines conditions: `must` is AND, `should` is OR, `must_not` is NOT. Each condition is a `FieldCondition` on a payload key.
**2. In plain words:** `must n_words <= 30` and `must_not doc_id == shipping_and_delivery`.
**3. Example to quote:** > "Filters are applied inside the search, not after it, so I still get `limit` results."
**Seen in:** `main.py`, part E

## FieldCondition, MatchValue, MatchAny, Range  `(class)`
**1. Definition:** `FieldCondition(key, match=...)` tests one payload field. `MatchValue` equals one value, `MatchAny` is in a list, `Range(gt, gte, lt, lte)` gives numeric bounds.
**2. In plain words:** `MatchAny(any=['loyalty_program','customer_support'])` restricts the search to those two docs.
**3. Example to quote:** > "Exact match, IN-list and range cover most RAG metadata filters."
**Seen in:** `main.py`, part E

## query_points  `(function)`
**1. Definition:** `client.query_points(collection, query=vector, limit, query_filter, score_threshold, with_payload)` returns a response whose `.points` are scored points (id, score, payload). It replaces the deprecated `client.search`.
**2. In plain words:** Top-1 for the delivery question is `shipping_and_delivery#Tracking an order` at 0.471.
**3. Example to quote:** > "query_points is the current unified query API; search is gone in newer clients."
**Seen in:** `main.py`, part E

## score_threshold  `(syntax)`
**1. Definition:** A query parameter that drops hits whose score is worse than the cut-off (for Cosine, lower than it).
**2. In plain words:** For an unanswerable question at 0.45 the list can be empty, which is the refusal signal for step 08.
**3. Example to quote:** > "An empty result list is a cheap signal to say I don't know."
**Seen in:** `main.py`, part E

## with_payload and with_vectors  `(syntax)`
**1. Definition:** Flags (bool or list of field names) controlling whether payload and vectors are returned with points.
**2. In plain words:** `with_payload=['chunk_id']` returns only that field; `with_vectors=False` avoids shipping 384 floats per point.
**3. Example to quote:** > "I return only the payload fields I need to keep responses small."
**Seen in:** `main.py`, parts E and F

## retrieve (get by id)  `(function)`
**1. Definition:** `client.retrieve(collection, ids, with_payload, with_vectors)` fetches specific points by id without any similarity search.
**2. In plain words:** Fetches the 'Delivery times' point via its recomputed uuid5.
**3. Example to quote:** > "Deterministic ids let me look up a chunk directly."
**Seen in:** `main.py`, part F

## scroll (pagination)  `(function)`
**1. Definition:** `client.scroll(collection, limit, offset, with_payload, with_vectors)` pages through points without a query; returns `(points, next_page_offset)`, where the offset is None at the end.
**2. In plain words:** `limit=10` returns 10 points and a non-None offset.
**3. Example to quote:** > "Scroll is how I iterate or export a whole collection, ten at a time."
**Seen in:** `main.py`, part F

## Delete by filter (FilterSelector)  `(function)`
**1. Definition:** `client.delete(collection, points_selector=models.FilterSelector(filter=...))` removes every point matching a filter; `PointIdsList` removes given ids.
**2. In plain words:** Deleting `doc_id == privacy_and_data` shrinks the count, mirroring a document taken down.
**3. Example to quote:** > "Because every chunk carries doc_id, I can remove a whole document in one call."
**Seen in:** `main.py`, part F

## count  `(function)`
**1. Definition:** `client.count(collection).count` returns the number of points (exact by default, optionally with a filter).
**2. In plain words:** Used to show 36 after the first and second upsert.
**3. Example to quote:** > "I assert counts after ingestion to catch duplicates."
**Seen in:** `main.py`, parts C and F

## Client modes (:memory:, path, url)  `(concept)`
**1. Definition:** `QdrantClient(':memory:')` runs in-process and loses data on exit; `QdrantClient(path=...)` runs in-process and persists to a folder with a lock; `QdrantClient(url='http://localhost:6333')` talks to a server over HTTP/gRPC.
**2. In plain words:** Same API in all three, so tests use memory and production uses a server.
**3. Example to quote:** > "I develop against :memory: and ship against a server; the code does not change."
**Seen in:** `main.py`, part A

## Local mode vs server mode  `(concept)`
**1. Definition:** Local mode is a Python implementation for dev, tests and small data (single process). The server is the Rust service with HNSW, real payload indexes, replication, sharding and snapshots.
**2. In plain words:** Local mode ignores `create_payload_index`, and `close()` is needed to release the folder lock before reopening.
**3. Example to quote:** > "Local mode is for prototypes; real traffic goes to the server."
**Seen in:** `main.py`, parts A and F

## Named vectors  `(concept)`
**1. Definition:** A collection can store several vectors per point, each with its own name, size and distance (`vectors_config={'dense': VectorParams(...), ...}`), plus sparse vectors for hybrid search.
**2. In plain words:** Our collection uses a single unnamed vector, so queries need no `using=` argument. Named vectors are the route to dense + sparse hybrid.
**3. Example to quote:** > "One point, several embeddings: name them and choose with using=."
**Seen in:** background only, not used in `main.py`

## Quantization (in Qdrant)  `(concept)`
**1. Definition:** Compressing stored vectors to save memory and speed search, via `quantization_config`: scalar (float32 to int8, about 4x), binary (about 32x) or product quantization, usually with rescoring against the original vectors.
**2. In plain words:** Not enabled here. Worth turning on only for millions of vectors; the math is in module 21 step 04.
**3. Example to quote:** > "Quantization trades a little recall for 4x to 32x less RAM, and rescoring recovers most of it."
**Seen in:** background only, not used in `main.py`

## qdrant_client.models  `(class)`
**1. Definition:** The module of typed request/response classes (`PointStruct`, `Filter`, `VectorParams`, `Distance`, ...) that the client serialises to REST/gRPC.
**2. In plain words:** `from qdrant_client import QdrantClient, models`.
**3. Example to quote:** > "The models module holds schemas for every Qdrant request."
**Seen in:** `main.py`, imports

## Collection lifecycle calls  `(function)`
**1. Definition:** `collection_exists`, `create_collection`, `get_collection` (read back `info.config.params.vectors.size`) and `delete_collection`.
**2. In plain words:** Makes the script re-runnable: delete `kb` if it exists, then create it.
**3. Example to quote:** > "I guard create with collection_exists so reruns are safe."
**Seen in:** `main.py`, part B

## tempfile.TemporaryDirectory  `(function)`
**1. Definition:** Standard library: creates a temp folder, removed by `.cleanup()`.
**2. In plain words:** Used as the on-disk home of the local-mode database, then reopened to prove persistence.
**3. Example to quote:** > "A temp dir keeps the demo from leaving files behind."
**Seen in:** `main.py`, parts A and F
