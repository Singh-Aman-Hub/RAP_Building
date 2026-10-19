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
