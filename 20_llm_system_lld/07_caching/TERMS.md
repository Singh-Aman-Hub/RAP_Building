# Terms — Caching: exact, semantic, TTL, LRU, invalidation, and provider-side prompt caching

Folder: `20_llm_system_lld/07_caching` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [20 README](../README.md)

**Terms in this folder:** [Exact-match cache (cache key)](#exact-match-cache-cache-key--concept) · [Semantic cache](#semantic-cache--concept) · [TTL (time to live)](#ttl-time-to-live--concept) · [LRU eviction](#lru-eviction--concept) · [Sentinel value (MISS)](#sentinel-value-miss--syntax) · [Cache scope (versioned key)](#cache-scope-versioned-key--concept) · [Cache invalidation](#cache-invalidation--concept) · [Provider-side prompt caching (prefix cache)](#provider-side-prompt-caching-prefix-cache--concept) · [Hit rate (cache statistics)](#hit-rate-cache-statistics--concept) · [ACL leak through caches](#acl-leak-through-caches--concept) · [Truthiness of containers (__len__)](#truthiness-of-containers-__len__--syntax) · [Monkey patching](#monkey-patching--syntax)

---

## Exact-match cache (cache key)  `(concept)`
**1. Definition:** A cache whose key is a hash of the normalised prompt, sampling parameters and scope; only identical requests hit.
**2. In plain words:** A phone's recent-calls shortcut: same number, same entry.
**3. Example to quote:** > "Hash everything that can change the answer, including temperature and index version."
```python
cache_key(system, user, params, scope)
```
**Seen in:** `ragkit/cache.py`

## Semantic cache  `(concept)`
**1. Definition:** A cache that returns a stored answer when the new question's embedding is within a similarity threshold of a stored question's.
**2. In plain words:** A librarian who recognises 'same question, different words'.
**3. Example to quote:** > "Similarity is not equivalence: 'is shipping not free' scored 0.93 against 'is shipping free'."
**Seen in:** `ragkit/cache.py`

## TTL (time to live)  `(concept)`
**1. Definition:** The number of seconds after which a cache entry is considered stale and discarded.
**2. In plain words:** A best-before date.
**3. Example to quote:** > "TTL bounds staleness when you cannot detect changes."
```python
expires = now + ttl
```
**Seen in:** `ragkit/cache.py`

## LRU eviction  `(concept)`
**1. Definition:** When a cache is full, remove the entry that was used least recently.
**2. In plain words:** Throwing out the book you have not opened longest.
**3. Example to quote:** > "OrderedDict.move_to_end on every hit gives O(1) LRU."
```python
self._data.popitem(last=False)
```
**Seen in:** `ragkit/cache.py`

## Sentinel value (MISS)  `(syntax)`
**1. Definition:** A unique object used to mean 'not found', so that None can be a stored value.
**2. In plain words:** A special 'empty' token that nobody could confuse with data.
**3. Example to quote:** > "`MISS = object()`; compare with `is`."
```python
if hit is not MISS: return hit
```
**Seen in:** `ragkit/cache.py`

## Cache scope (versioned key)  `(concept)`
**1. Definition:** The part of the key (index version, model, tenant) that makes entries from other contexts unreachable.
**2. In plain words:** A room number in a hotel: same bed number, different room.
**3. Example to quote:** > "Changing the index version invalidates every old entry by construction."
**Seen in:** `ragkit/cache.py`

## Cache invalidation  `(concept)`
**1. Definition:** Ensuring entries that no longer reflect the truth are not served.
**2. In plain words:** Updating the menu after the recipe changes.
**3. Example to quote:** > "Versioned keys, TTL, explicit purge, event-driven purge."
**Seen in:** `WALKTHROUGH.md`, theory

## Provider-side prompt caching (prefix cache)  `(concept)`
**1. Definition:** The API reuses the processed state of an identical prompt prefix marked with cache_control; reads cost far less than normal input.
**2. In plain words:** A bookmark in a long document the API has already read.
**3. Example to quote:** > "Put stable content first; a timestamp at the top kills hits. Writes cost 1.25x, reads about 0.1x (check the model's pricing)."
```python
{"cache_control": {"type": "ephemeral"}}
```
**Seen in:** `WALKTHROUGH.md`, theory

## Hit rate (cache statistics)  `(concept)`
**1. Definition:** hits / (hits + misses); with evictions and expirations it tells whether a cache is useful and sized right.
**2. In plain words:** How often the shortcut was used.
**3. Example to quote:** > "Emit it as a metric (step 09)."
```python
stats.hit_rate
```
**Seen in:** `ragkit/cache.py`

## ACL leak through caches  `(concept)`
**1. Definition:** Serving a cached answer built from documents the current user may not read.
**2. In plain words:** Showing one customer another's receipt.
**3. Example to quote:** > "Put the permission group in the cache scope."
**Seen in:** `WALKTHROUGH.md`, theory

## Truthiness of containers (__len__)  `(syntax)`
**1. Definition:** Objects defining __len__ are falsy when empty, so `if obj:` can mean 'non-empty' rather than 'exists'.
**2. In plain words:** An empty box looks like no box.
**3. Example to quote:** > "Use `is not None` for optional collaborators."
```python
if self.semantic is not None:
```
**Seen in:** `ragkit/cache.py`

## Monkey patching  `(syntax)`
**1. Definition:** Replacing an attribute or method of an object at runtime.
**2. In plain words:** Swapping a part while the machine runs.
**3. Example to quote:** > "main.py counts pipeline runs by replacing `answer`; fine in a demo, avoid in production code."
```python
inner.answer = counting_answer
```
**Seen in:** `main.py`
