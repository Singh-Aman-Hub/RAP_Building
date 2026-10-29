# Hints: 02 Sentence Transformers

Open one hint at a time.

## First move
<details><summary>Hint 0</summary>

Inside `search()`, print `type(x)` and `x.shape` for `query_embedding`, `body_embeddings`, `scores` and `top`. Write down what you *expected* each shape to be before you look.
</details>

## Bug A: "only integer tensors of a single element can be converted to an index"
<details><summary>Hint A1</summary>

Look at the shape of `scores`. How many dimensions does it have? Check the `util.cos_sim` docstring (`help(util.cos_sim)`) to see what shape it returns for two inputs.
</details>
<details><summary>Hint A2</summary>

If `scores` is 2-D, what is `top` after `np.argsort(scores)[:k]`? What does the loop variable `i` contain on the first iteration?
</details>

## Bug B: the "top" results are the worst ones
<details><summary>Hint B1</summary>

Print `scores` next to `top`. Does `top` list the indexes of the highest scores or the lowest?
</details>
<details><summary>Hint B2</summary>

Check which direction `np.argsort` sorts in by default, and how to get the opposite order.
</details>

## Bug C: "Object of type ... is not JSON serializable"
<details><summary>Hint C1</summary>

Print `type(scores[i])`. The `json` module only knows built-in Python types.
</details>
<details><summary>Hint C2</summary>

The fix belongs in `search()`, not in a custom JSON encoder. The docstring says the result must be JSON-serialisable.
</details>

## Bug D: titles don't match their snippets
<details><summary>Hint D1</summary>

Compare `len(titles)` with `len(bodies)`. Then print `bodies` in two separate runs of the script. Is the order the same each time?
</details>
<details><summary>Hint D2</summary>

`titles` and `bodies` are *parallel lists*: index `i` in one must describe the same article as index `i` in the other. Which line breaks that guarantee, and in how many ways? (Hint: both length and order.)
</details>
<details><summary>Hint D3</summary>

Look for a way to drop duplicates that keeps the original order and keeps each title attached to its body. Treat the `(title, body)` pair as the thing you deduplicate.
</details>

## Verification ideas
- `assert len(titles) == len(bodies) == len(body_embeddings)`
- Run the script 3 times. The output must be identical each time.
- `assert all(r["score"] >= s["score"] for r, s in zip(res, res[1:]))` (scores are sorted, highest first).
- `json.dumps(search("anything"))` must not raise.
