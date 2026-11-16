# Step 05 — Databases and model storage: the prediction log

## The story so far
Step 04 told us which model is live. What we still lack is a record of what it said and to whom.

## The pain point
Without a record you cannot debug a bad answer, compare two versions, or collect the real examples needed for the next training set.

## The spark
Write one row per prediction (time, model version, input, label, confidence, latency) into a SQL table, using parameterized queries so user text can never become SQL.

## Goal
Persist every prediction (input, label, confidence, latency, **model version**) in SQLite and answer operational questions with SQL.

## Run it
```bash
cd 10_ml_system_architecture/05_storage_and_prediction_log && python main.py
```
Real output (latency numbers vary):
```
label counts   : {'positive': 30, 'negative': 10}
latency stats  : {'count': 40, 'avg': 1.364, 'p50': 0.302, 'p95': 11.217, 'max': 24.496}
low confidence :
    {'id': 3, 'input_text': 'It arrived Tuesday', 'label': 'positive', 'confidence': 0.526}
    ...
unsafe f-string SQL matches every row: 40 rows; parameterized query matches: 0
```
Look at `p95` (11 ms) versus `p50` (0.3 ms): the first calls are slow (warm-up). **Averages hide tails** — this is why we track percentiles (step 07).

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Prediction log | inference log, request log, serving log | Near-synonyms | A durable record of each prediction: input, output, version, latency. |
| SQLite | embedded SQL database, `sqlite3` module | No | A SQL database in one file or in memory; Postgres is the server-based production equivalent. |
| Parameterized query | prepared statement, bound parameters, `?` placeholders | Near-synonyms | Values travel separately from SQL text, preventing SQL injection. |
| SQL injection | SQLi | Same | User text turned into SQL by string-building; the opposite of using placeholders. |
| Latency percentiles | p50, p95, p99, tail latency | Related | p95 = 95% of requests are faster; averages hide slow tails. |
| Object storage vs database vs key-value store | S3/GCS vs Postgres vs Redis | No | Big files vs queryable rows vs fast lookups by key. |
| `commit` | persist, flush (loosely) | Not exactly | Makes pending changes permanent in the database. |

## Code, step by step

### 1. Docstring: why log predictions  (`prediction_log.py` lines 1–6)

```python
"""
prediction_log.py -- store every prediction in SQLite (a SQL database that lives in one file; built into Python).

Why log predictions?  Debugging ("what did the model say for this input?"), monitoring (drift, latency),
auditing, and building the NEXT training set from real traffic + user feedback.
"""
```

- **In simple words:** Says we keep every answer for debugging, monitoring, audits and future training.
- **Exactly:** Header text.
- **Why it is here:** Reasons to build the log.

### 2. Imports  (`prediction_log.py` lines 8–10)

```python
import sqlite3                       # built-in: no server to install
import time
from datetime import datetime, timezone
```

- **In simple words:** Tools for SQLite, timing and timestamps.
- **Exactly:** `sqlite3` ships with Python; no server needed.
- **Why it is here:** The whole database without installing anything.

### 3. SCHEMA: the table definition and an index  (`prediction_log.py` lines 12–23)

```python
SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,   -- unique row number, assigned by the database
    ts            TEXT    NOT NULL,                    -- when (UTC, ISO format)
    model_version TEXT    NOT NULL,                    -- WHICH model answered (needed for A/B and rollbacks)
    input_text    TEXT    NOT NULL,
    label         TEXT    NOT NULL,
    confidence    REAL    NOT NULL,
    latency_ms    REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_version ON predictions(model_version);  -- makes filtering by version fast
"""
```

- **In simple words:** The shape of the spreadsheet we will store answers in.
- **Exactly:** SQL: `CREATE TABLE IF NOT EXISTS predictions (...)` with columns id (auto-increment integer key), ts, model_version, input_text, label (TEXT), confidence and latency_ms (REAL), all `NOT NULL`. `CREATE INDEX` on `model_version` speeds filtering by version. `--` starts an SQL comment. Triple-quoted Python string run later by `executescript`.
- **Why it is here:** Without the table nothing can be saved; without NOT NULL bad rows slip in; model_version enables A/B comparison and rollbacks.

### 4. PredictionLog.__init__: open the database and create the table  (`prediction_log.py` lines 26–30)

```python
class PredictionLog:
    def __init__(self, path: str = ":memory:"):          # ":memory:" = temporary DB that vanishes on exit
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row               # rows behave like dicts: row["label"]
        self.conn.executescript(SCHEMA)
```

- **In simple words:** Connect to the database (in memory by default) and make the table.
- **Exactly:** `sqlite3.connect(":memory:")` makes a temporary DB. `row_factory = sqlite3.Row` lets rows be read by column name. `executescript` runs several SQL statements.
- **Why it is here:** In-memory is perfect for a demo; a path like `preds.db` gives a persistent file.

### 5. record(): insert one prediction row safely  (`prediction_log.py` lines 32–39)

```python
    def record(self, model_version, text, label, confidence, latency_ms) -> None:
        # The ? placeholders are PARAMETERIZED QUERIES: values are passed separately from the SQL text,
        # which prevents SQL injection. NEVER build SQL with f-strings from user input.
        self.conn.execute(
            "INSERT INTO predictions (ts, model_version, input_text, label, confidence, latency_ms) VALUES (?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), model_version, text, label, confidence, latency_ms),
        )
        self.conn.commit()                                # commit = make the change permanent
```

- **In simple words:** Write one answer into the table.
- **Exactly:** `INSERT ... VALUES (?,?,?,?,?,?)` with a 6-tuple: values are bound separately (parameterized). `datetime.now(timezone.utc).isoformat(timespec="seconds")` is the timestamp string. `commit()` makes it permanent.
- **Why it is here:** f-string SQL with user text is an injection hole; no commit means the row vanishes.

### 6. label_counts(): count predictions per label  (`prediction_log.py` lines 41–48)

```python
    def label_counts(self, model_version: str | None = None) -> dict:
        sql = "SELECT label, COUNT(*) AS n FROM predictions"
        args: tuple = ()
        if model_version:
            sql += " WHERE model_version = ?"
            args = (model_version,)
        sql += " GROUP BY label ORDER BY n DESC"
        return {row["label"]: row["n"] for row in self.conn.execute(sql, args)}
```

- **In simple words:** Ask 'how many of each answer?'
- **Exactly:** Builds SQL in steps; optional `WHERE model_version = ?`. `GROUP BY label ORDER BY n DESC` counts per label, biggest first. Returns dict. Real: `{'positive': 30, 'negative': 10}`.
- **Why it is here:** A skewed label mix is the first sign of trouble.

### 7. latency_stats(): average and percentiles  (`prediction_log.py` lines 50–55)

```python
    def latency_stats(self) -> dict:
        lat = sorted(r[0] for r in self.conn.execute("SELECT latency_ms FROM predictions"))
        if not lat:
            return {}
        pick = lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))], 3)     # nearest-rank percentile
        return {"count": len(lat), "avg": round(sum(lat) / len(lat), 3), "p50": pick(0.5), "p95": pick(0.95), "max": round(lat[-1], 3)}
```

- **In simple words:** Summarise how fast answers were.
- **Exactly:** Fetches all latencies, sorts. `pick(q)` = value at index `int(q*len)` (nearest rank). Real: `{'count': 40, 'avg': 2.448, 'p50': 1.298, 'p95': 3.606, 'max': 39.872}`. The max is far above p95: one slow outlier.
- **Why it is here:** Averages hide slow tails; percentiles are what SLOs use.

### 8. low_confidence(): the least certain predictions  (`prediction_log.py` lines 57–62)

```python
    def low_confidence(self, below: float = 0.6, limit: int = 5) -> list:
        """Rows worth a human look -- candidates for relabelling and the next training set."""
        cur = self.conn.execute(
            "SELECT id, input_text, label, confidence FROM predictions WHERE confidence < ? ORDER BY confidence LIMIT ?",
            (below, limit))
        return [dict(r) for r in cur]
```

- **In simple words:** Find the answers the model was least sure about.
- **Exactly:** `WHERE confidence < ? ORDER BY confidence LIMIT ?` with params `(below, limit)`; `dict(row)` converts each row to a plain dict.
- **Why it is here:** Those are the best candidates for human review and the next training set.

### 9. Docstring: goal and storage map  (`main.py` lines 1–10)

```python
"""
GOAL:   Store predictions in a database and ask useful questions about them with SQL.
        Also see where each kind of data lives in an ML system (the "storage map").

RUN:    cd 10_ml_system_architecture/05_storage_and_prediction_log && python main.py

EXPECTED OUTPUT (shape): label counts, latency stats, low-confidence rows, an injection demo, a storage map.

NEEDS:  scikit-learn, joblib
"""
```

- **In simple words:** What this demo shows.
- **Exactly:** Header comment.
- **Why it is here:** Orientation.

### 10. Imports, reusing step 02's inference code  (`main.py` lines 12–20)

```python
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_training_vs_inference_pipelines"))
from predict import load_artifact, predict_text   # noqa: E402  (reuses step 02's inference code)
from train import train                           # noqa: E402

from prediction_log import PredictionLog
```

- **In simple words:** Re-use the earlier model code instead of rewriting.
- **Exactly:** `sys.path.insert` makes step 02's `predict` and `train` importable; `PredictionLog` from prediction_log.py in this folder.
- **Why it is here:** Shows that one artifact feeds many consumers.

### 11. Make sure a model exists and open the log  (`main.py` lines 22–26)

```python
if __name__ == "__main__":
    if not (Path(__file__).resolve().parents[1] / "02_training_vs_inference_pipelines/artifacts/model.joblib").exists():
        train()                                   # make sure an artifact exists
    model, meta = load_artifact()
    log = PredictionLog()                         # in-memory DB for the demo; pass "preds.db" to keep a file
```

- **In simple words:** Train once if no model file; load it; start an in-memory DB.
- **Exactly:** `train()` is only called if `artifacts/model.joblib` is missing.
- **Why it is here:** Makes the demo runnable from a fresh clone.

### 12. Simulate 40 requests and log each one  (`main.py` lines 28–34)

```python
    traffic = ["Great hotel", "Awful delivery", "It arrived Tuesday", "The app is brilliant", "Meh, it exists",
               "Terrible, worthless game", "Lovely teacher", "I bought a phone"] * 5
    for text in traffic:
        t0 = time.perf_counter()
        result = predict_text(model, text, threshold=0.0)         # threshold 0.0: always return the raw best label
        ms = (time.perf_counter() - t0) * 1000
        log.record("v-demo", text, result["label"], result["confidence"], ms)
```

- **In simple words:** Pretend traffic arrives; predict and record each.
- **Exactly:** 8 texts x 5 = 40 requests. `threshold=0.0` means the label is never 'uncertain'. Latency is measured with `perf_counter`. `log.record('v-demo', ...)` stores one row.
- **Why it is here:** Gives the log real contents to query.

### 13. Query the log: counts, latency, low confidence  (`main.py` lines 36–40)

```python
    print("label counts   :", log.label_counts())
    print("latency stats  :", log.latency_stats())
    print("low confidence :")
    for row in log.low_confidence(below=0.6, limit=3):
        print("   ", row)
```

- **In simple words:** Ask three questions.
- **Exactly:** Real: `{'positive': 30, 'negative': 10}`; latency stats above; low-confidence rows e.g. `{'id': 3, 'input_text': 'It arrived Tuesday', 'label': 'positive', 'confidence': 0.526}`. 'positive' 30 of 40 because some texts are neutral and the model must pick a side.
- **Why it is here:** Shows what monitoring and data-curation queries look like.

### 14. SQL injection demo: f-string vs placeholder  (`main.py` lines 42–47)

```python
    # SQL injection demo: why we use ? placeholders.
    evil = "x' OR '1'='1"
    unsafe_sql = f"SELECT COUNT(*) FROM predictions WHERE label = '{evil}'"      # BAD: user text becomes SQL
    safe_count = log.conn.execute("SELECT COUNT(*) FROM predictions WHERE label = ?", (evil,)).fetchone()[0]
    print(f"\nunsafe f-string SQL matches every row: {log.conn.execute(unsafe_sql).fetchone()[0]} rows;"
          f" parameterized query matches: {safe_count}")
```

- **In simple words:** Same malicious text, two ways of using it.
- **Exactly:** `evil = "x' OR '1'='1"` closes the quote and adds an always-true condition. The f-string query becomes `WHERE label = 'x' OR '1'='1'` and matches all rows: real `40 rows`. With `?` the text is treated as a plain value and matches `0`.
- **Why it is here:** Shows why you never build SQL with f-strings from user input.

### 15. Print the storage map  (`main.py` lines 49–56)

```python
    print("""
STORAGE MAP -- what lives where in a typical ML system
  raw / training data ........ object storage (S3, GCS) or a data warehouse     (big, append-only)
  model artifacts ............ object storage + registry (step 04)              (immutable, versioned)
  online features ............ key-value store (Redis) -- fast lookups            (low latency)
  prediction logs / feedback . relational DB (Postgres; SQLite here)             (queryable, durable)
  metrics .................... time-series DB (Prometheus)                       (dashboards, alerts)
""")
```

- **In simple words:** Where each kind of data lives.
- **Exactly:** Multi-line string: raw data in object storage/warehouse, models in object storage + registry, online features in Redis, prediction logs in a relational DB, metrics in a time-series DB like Prometheus.
- **Why it is here:** Different data has different access patterns, so different stores.

### 16. Comments: things to try and break  (`main.py` lines 57–59)

```python
    # TRY IT: change ":memory:" to "preds.db", run twice, and check that counts accumulate. Then delete the file.
    # TRY IT: write a method that returns average confidence per label.
    # BREAK IT: rename a column in a query to 'lable' and read the sqlite3.OperationalError.
```

- **In simple words:** Experiments.
- **Exactly:** Comments only.
- **Why it is here:** Pointers.


## The storage map (what lives where)
| Data | Typical store | Why |
|---|---|---|
| Raw/training data | object storage (S3/GCS), warehouse | big, append-only, cheap |
| Model artifacts | object storage + registry (step 04) | immutable, versioned |
| Online features | Redis / key-value | millisecond lookups |
| Predictions + feedback | relational DB (Postgres; SQLite here) | queryable, durable, joinable with outcomes |
| Metrics | Prometheus / time-series | dashboards, alerts |
| Logs | stdout → log aggregator | search, debugging |

## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `CREATE TABLE IF NOT EXISTS ...` | SQL: define a table once | — |
| `INSERT ... VALUES (?,?,?)` + tuple | **Parameterized query**: values travel separately from the SQL | `execute("... WHERE id=?", (5,))` |
| `conn.commit()` | Make changes permanent (a transaction ends) | — |
| `sqlite3.Row` row factory | Rows accessible by column name | `row["label"]` |
| `GROUP BY label` + `COUNT(*)` | Aggregate rows per label | — |
| `dict(row)`, `{r["a"]: r["b"] for r in cur}` | Convert rows to dicts | — |
| `str \| None = None` default argument | Optional parameter | — |

## Libraries & functions
| Name | Kind | Does |
|---|---|---|
| `sqlite3.connect(path)` | stdlib | Open/create a DB file (`":memory:"` = temporary) |
| `conn.execute(sql, params)` | stdlib | Run one statement; returns a cursor you can iterate |
| `conn.executescript(sql)` | stdlib | Run several statements (schema setup) |
| `CREATE INDEX` | SQL | Makes filtering by a column fast |

## Execution trace
`log.record("v-demo", "Great hotel", "positive", 0.78, 0.4)` → INSERT with 6 bound values → commit → row id 1. Later `label_counts()` runs `SELECT label, COUNT(*) ... GROUP BY label ORDER BY n DESC` → `{'positive': 30, 'negative': 10}`.

## SQL injection in one demo
`"... WHERE label = '" + user_text + "'"` with `user_text = "x' OR '1'='1"` becomes `WHERE label = 'x' OR '1'='1'` → matches all rows (an attacker could do worse). With `?` placeholders the whole string is treated as a *value*, matching nothing. **Never build SQL with f-strings from user input.**

## Experiments
1. Use `PredictionLog("preds.db")`, run twice → counts accumulate (then delete the file).
2. Write `avg_confidence_by_label()` with `AVG(confidence)`.
3. Log `model_version` as `v1` for half the traffic and `v2` for the rest; call `label_counts("v2")`.

## Debugging practice
Rename a column in a query (`lable`) → `sqlite3.OperationalError: no such column: lable`. Check names against the `CREATE TABLE`. Open the DB in the `sqlite3` CLI: `.schema`, `SELECT * FROM predictions LIMIT 5;`.

## Rebuild checklist
- [ ] Create a table, insert with `?`, query with `GROUP BY`.
- [ ] Explain what to log (and what **not** to: PII, secrets).
- [ ] Explain why the log includes `model_version`.

## The new problem (cliffhanger)
The log answers one request at a time, but is answering one at a time the right design for every job? Some work is cheaper in bulk. Next: [06_batch_vs_realtime](../06_batch_vs_realtime/).

## Say it aloud
> **Say it aloud (interview box):** Every prediction is logged with timestamp, input, output, confidence, latency and model version in a relational store, which powers debugging, monitoring and the next training set. I always use parameterized queries to avoid SQL injection. The cost is that raw inputs may contain personal data and add write load, so I sample, mask, or log lengths instead.
