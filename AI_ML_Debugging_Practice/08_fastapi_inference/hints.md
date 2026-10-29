# Hints: 08 FastAPI Inference Service

Open one hint at a time.

## Bug A: 422 "Field required" on valid requests
<details><summary>Hint A1</summary>

Read the 422 body carefully. `loc` tells you which field FastAPI was looking for, and `input` shows what it received.
</details>
<details><summary>Hint A2</summary>

Compare the Pydantic model with the request shape in the contract: the field's name **and** its type. Look at how the endpoint body uses the request, too.
</details>

## Bug B: 500 Internal Server Error
<details><summary>Hint B1</summary>

Temporarily switch the `TestClient` to `raise_server_exceptions=True`, or run the app with `uvicorn` and watch the terminal. You need the real traceback, not the 500 status.
</details>
<details><summary>Hint B2</summary>

What type is `vectors`? FastAPI turns return values into JSON with `jsonable_encoder`. Which Python types does JSON understand?
</details>

## Bug C: slow requests, and `model_loaded` is false
<details><summary>Hint C1</summary>

Add a `print` at the start of `embed()` and inside it, around the line that creates the model. How many times is the model created during the smoke test?
</details>
<details><summary>Hint C2</summary>

There is a module-level `model = None`, and a variable called `model` inside `embed()`. Are they the same variable? Look up Python's scoping rules for assignment inside a function.
</details>
<details><summary>Hint C3</summary>

The fix is about **when** the model is created, not about adding `global` inside the endpoint. Read FastAPI's "Lifespan Events" docs page.
</details>

## Bug D: an empty list causes a 500
<details><summary>Hint D1</summary>

After the other fixes, look at the traceback for the empty-list call. What does `model.encode([])` return, and what does `.shape[1]` do on it?
</details>
<details><summary>Hint D2</summary>

The constraint says to use Pydantic for validation. Pydantic can enforce a minimum length on a list field (see `Field` in the Pydantic docs, or `conlist`).
</details>

## Verification ideas
- `assert response.json()["dim"] == len(response.json()["embeddings"][0]) == 384`
- Time 10 warm calls and assert the median is well below the first call's time.
- `assert client.post("/embed", json={"texts": []}).status_code == 422`
- Open `/docs` (with uvicorn). Does the generated schema match the contract?
