# Hints: 07 Similarity Search and Filtering

Open one hint at a time.

## First move
<details><summary>Hint 0</summary>

Debug filters one condition at a time. Try the search with **no filter**, then with only the department condition, then with only the year condition. `client.count(COLLECTION, count_filter=...)` tells you how many points match a filter without any vectors involved.
</details>

## Bug A: the two conditions combine the wrong way
<details><summary>Hint A1</summary>

Read Qdrant's *Concepts → Filtering* page, section "Filtering clauses". Which clause means "all of these must hold"?
</details>
<details><summary>Hint A2</summary>

Once the other bugs are fixed, a symptom of this one is documents from the *wrong department* or the *wrong year* appearing in results.
</details>

## Bug B: the year condition never matches
<details><summary>Hint B1</summary>

Look at a stored point: `client.scroll(COLLECTION, limit=1, with_payload=True)`. Print `type(payload["year"])`.
</details>
<details><summary>Hint B2</summary>

What types does `csv.DictReader` produce for every field? What does a `Range` condition compare against?
</details>

## Bug C: the department condition never matches
<details><summary>Hint C1</summary>

Compare the department value stored in the payload with the value sent by the dropdown, character by character.
</details>
<details><summary>Hint C2</summary>

`MatchValue` is an exact keyword match. Decide on **one canonical form**, then apply it in *both* places where department values enter the system.
</details>

## Bug D: `'NoneType' object is not subscriptable`
<details><summary>Hint D1</summary>

This appears as soon as any result comes back. Print `hits[0]`. Which field is `None`?
</details>
<details><summary>Hint D2</summary>

Check the arguments passed to `query_points` (`help(QdrantClient.query_points)`). One of them decides whether the payload comes back with each hit.
</details>

## Verification ideas
- For every returned tuple, `assert department.lower() == requested.lower() and year >= min_year`.
- `search(..., department="HR", ...)` and `search(..., department="hr", ...)` return identical lists.
- `assert all(isinstance(p.payload["year"], int) for p in client.scroll(COLLECTION, limit=100)[0])`
