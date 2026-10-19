# Terms — Step 05: Storage and the prediction log

Folder: `10_ml_system_architecture/05_storage_and_prediction_log` · Files covered: `main.py`, `prediction_log.py`, `WALKTHROUGH.md`
Back to module: [10 ML system architecture](../README.md)

**Terms in this folder:** [Prediction log](#prediction-log-concept) · [Storage map](#storage-map-concept) · [SQLite](#sqlite-standard-library) · [sqlite3.connect(path)](#sqlite3connectpath-standard-library) · [sqlite3.Row (row_factory)](#sqlite3row-row_factory-standard-library) · [CREATE TABLE IF NOT EXISTS](#create-table-if-not-exists-sql) · [PRIMARY KEY AUTOINCREMENT](#primary-key-autoincrement-sql) · [CREATE INDEX](#create-index-sql) · [executescript vs execute](#executescript-vs-execute-standard-library) · [Parameterized query](#parameterized-query-concept) · [SQL injection](#sql-injection-concept) · [commit](#commit-standard-library) · [INSERT ... VALUES (?,?,?)](#insert-values-sql) · [GROUP BY and COUNT(*)](#group-by-and-count-sql) · [Percentiles (p50, p95)](#percentiles-p50-p95-concept) · [Nearest-rank percentile with lambda](#nearest-rank-percentile-with-lambda-syntax) · [sorted() with generator](#sorted-with-generator-python-built-in) · [Dict comprehension from rows](#dict-comprehension-from-rows-syntax) · [Optional parameter (str | None = None)](#optional-parameter-str-none-none-syntax) · [Low-confidence rows](#low-confidence-rows-concept) · [PII and what not to log](#pii-and-what-not-to-log-concept) · [sqlite3.OperationalError](#sqlite3operationalerror-standard-library) · [Reusing inference code from step 02](#reusing-inference-code-from-step-02-concept) · [threshold=0.0 for raw labels](#threshold00-for-raw-labels-concept) · [Warm-up latency](#warm-up-latency-concept) · [Repeating a list with *](#repeating-a-list-with-syntax) · [Docstring-level design: why log predictions](#docstring-level-design-why-log-predictions-concept)

---

## Prediction log  `(concept)`
**1. Definition:** A persistent record of every prediction the service makes, including time, model version, input, output, confidence and latency.
**2. In plain words:** It lets you answer "what did the model say for that input?" weeks later, monitor drift and latency, audit decisions, and mine real traffic for the next training set. Logging the model version is essential for rollbacks and A/B tests.
**3. Example to quote:**
> "Each prediction writes a row to `predictions` with `model_version`, so when a user complains I can see exactly which model answered and how confident it was."

**Seen in:** `prediction_log.py`

## Storage map  `(concept)`
**1. Definition:** A summary of where each kind of ML data usually lives: object storage for raw data and artifacts, key-value stores for online features, relational databases for prediction logs, time-series databases for metrics, and a log aggregator for logs.
**2. In plain words:** Different data has different needs (size, speed, queries), so there is no single database. Knowing the map lets you answer system-design questions confidently.
**3. Example to quote:**
> "Raw data in S3, model files in a registry on object storage, online features in Redis, predictions in Postgres, metrics in Prometheus."

**Seen in:** `main.py` print, `WALKTHROUGH.md`

## SQLite  `(standard library)`
**1. Definition:** A serverless SQL database stored in a single file, available in Python through the built-in `sqlite3` module.
**2. In plain words:** There is nothing to install or run, which makes it ideal for demos and small services. For many concurrent writers a server database such as Postgres is the normal choice.
**3. Example to quote:**
> "I use SQLite for the prediction log here, and I'd move to Postgres when many services write at once."

**Seen in:** `prediction_log.py`

## sqlite3.connect(path)  `(standard library)`
**1. Definition:** Opens (or creates) a SQLite database; the special path `":memory:"` creates a temporary database that disappears when the program exits.
**2. In plain words:** `PredictionLog()` defaults to memory for demos. Passing `"preds.db"` keeps the data in a file, so counts accumulate across runs.
**3. Example to quote:**
> "`sqlite3.connect(':memory:')` is a throwaway database; with `'preds.db'` my log survives restarts."

```python
import sqlite3
conn = sqlite3.connect(":memory:")
```

**Seen in:** `PredictionLog.__init__`

## sqlite3.Row (row_factory)  `(standard library)`
**1. Definition:** Setting `conn.row_factory = sqlite3.Row` makes query rows accessible by column name, like `row["label"]`.
**2. In plain words:** Without it, rows are plain tuples and you must remember column positions. With it the code is readable and `dict(row)` converts a row to a dict.
**3. Example to quote:**
> "`{row['label']: row['n'] for row in cursor}` is readable because rows support column names."

**Seen in:** `PredictionLog.__init__`, `label_counts`

## CREATE TABLE IF NOT EXISTS  `(SQL)`
**1. Definition:** A SQL statement that defines a table with typed columns and constraints, doing nothing if the table already exists.
**2. In plain words:** It makes setup safe to run every time the program starts. Constraints like `NOT NULL` stop incomplete rows from being stored.
**3. Example to quote:**
> "My schema creates `predictions` with id, ts, model_version, input_text, label, confidence and latency_ms, all required."

**Seen in:** `SCHEMA`

## PRIMARY KEY AUTOINCREMENT  `(SQL)`
**1. Definition:** A column constraint that makes each row's `id` unique and lets the database assign increasing numbers automatically.
**2. In plain words:** You never choose ids yourself; the database gives each prediction its own number, so rows can be referenced or joined later.
**3. Example to quote:**
> "`id INTEGER PRIMARY KEY AUTOINCREMENT` means the first prediction is row 1, the next row 2, with no code from me."

**Seen in:** `SCHEMA`

## CREATE INDEX  `(SQL)`
**1. Definition:** Builds a lookup structure on a column so queries filtering by that column are fast.
**2. In plain words:** Without an index the database scans every row to find a model version. With millions of predictions an index on `model_version` keeps queries quick, at the cost of slightly slower writes and more storage.
**3. Example to quote:**
> "`CREATE INDEX idx_pred_version ON predictions(model_version)` makes 'show me everything v2 predicted' fast."

**Seen in:** `SCHEMA`

## executescript vs execute  `(standard library)`
**1. Definition:** `conn.executescript(sql)` runs several SQL statements at once (used for schema setup); `conn.execute(sql, params)` runs one statement and returns a cursor.
**2. In plain words:** Use `executescript` for creating tables and `execute` with parameters for normal queries and inserts. The cursor can be iterated to read rows.
**3. Example to quote:**
> "I create the table and index with `executescript(SCHEMA)` and insert rows with `execute(..., params)`."

**Seen in:** `PredictionLog`

## Parameterized query  `(concept)`
**1. Definition:** A SQL statement with `?` placeholders where values are supplied separately as a tuple, so the database treats them strictly as data, never as SQL code.
**2. In plain words:** It is the main defence against SQL injection. Never build SQL with f-strings from user input; always pass values separately.
**3. Example to quote:**
> "`execute('SELECT COUNT(*) FROM predictions WHERE label = ?', (evil,))` matched 0 rows, while the f-string version matched all 40."

```python
conn.execute("SELECT * FROM t WHERE id = ?", (5,))
```

**Seen in:** `record`, `main.py` injection demo

## SQL injection  `(concept)`
**1. Definition:** An attack where user-controlled text is concatenated into a SQL statement and changes its meaning.
**2. In plain words:** Putting `x' OR '1'='1` into `WHERE label = '...'` makes the condition always true, so the query returns everything; worse payloads can delete data. Placeholders make the whole string a harmless value.
**3. Example to quote:**
> "My demo's unsafe f-string query matched every row, while the parameterized one matched none."

**Seen in:** `main.py`

## commit  `(standard library)`
**1. Definition:** `conn.commit()` ends the current transaction and makes its changes permanent.
**2. In plain words:** Without a commit the insert may be lost when the connection closes. In `record` each prediction is committed right after insertion.
**3. Example to quote:**
> "After each `INSERT` I call `self.conn.commit()` so the row is saved for real."

**Seen in:** `record`

## INSERT ... VALUES (?,?,?)  `(SQL)`
**1. Definition:** SQL to add a row, with one placeholder per column value.
**2. In plain words:** The tuple passed to `execute` supplies the values in the same order as the listed columns.
**3. Example to quote:**
> "My insert lists six columns and passes a six-item tuple: timestamp, model version, text, label, confidence, latency."

**Seen in:** `record`

## GROUP BY and COUNT(*)  `(SQL)`
**1. Definition:** `GROUP BY label` collects rows with the same label together, and `COUNT(*)` counts rows in each group; `ORDER BY n DESC` sorts by count.
**2. In plain words:** It answers "how many positives vs negatives?" in one query. Counting labels over time is a basic way to spot drift.
**3. Example to quote:**
> "`SELECT label, COUNT(*) AS n ... GROUP BY label ORDER BY n DESC` returned `{'positive': 30, 'negative': 10}`."

**Seen in:** `label_counts`

## Percentiles (p50, p95)  `(concept)`
**1. Definition:** A percentile is the value below which a given percentage of observations fall; p50 is the median and p95 means 95% of requests were faster.
**2. In plain words:** Averages hide slow outliers. Here the median was 0.3 ms but p95 was about 11 ms because early calls were slow warm-ups. Users feel the tail, so you track p95 and p99, not just the average.
**3. Example to quote:**
> "Average latency looks fine, but p95 of 11 ms shows some requests are much slower, so I monitor percentiles."

**Seen in:** `latency_stats`

## Nearest-rank percentile with lambda  `(syntax)`
**1. Definition:** `pick = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))]` indexes into a sorted list at position `q * n`, capped at the last index.
**2. In plain words:** It computes a percentile without a library: sort the values, then pick the item at the right position. `min(...)` prevents an index error for q close to 1.
**3. Example to quote:**
> "`pick(0.95)` returns the value at 95% of the sorted latencies."

**Seen in:** `latency_stats`

## sorted() with generator  `(Python built-in)`
**1. Definition:** `sorted(r[0] for r in rows)` builds a sorted list from the first column of each row.
**2. In plain words:** Sorting the latencies is the first step to computing percentiles.
**3. Example to quote:**
> "I fetch all `latency_ms` values and sort them to compute p50 and p95."

**Seen in:** `latency_stats`

## Dict comprehension from rows  `(syntax)`
**1. Definition:** `{row["label"]: row["n"] for row in cursor}` creates a dict by iterating over query result rows.
**2. In plain words:** It converts database rows to a convenient Python dict like `{'positive': 30}`.
**3. Example to quote:**
> "`label_counts` returns a dict built straight from the cursor, one entry per label."

**Seen in:** `label_counts`

## Optional parameter (str | None = None)  `(syntax)`
**1. Definition:** A type hint and default meaning "a string or nothing"; code checks `if model_version:` to add an optional filter.
**2. In plain words:** Lets one method work for all versions or a specific one. The SQL string is built up in steps, with placeholders for any value.
**3. Example to quote:**
> "`label_counts('v2')` adds `WHERE model_version = ?`; with no argument it counts everything."

**Seen in:** `label_counts`

## Low-confidence rows  `(concept)`
**1. Definition:** Predictions whose confidence is below a threshold, selected with `WHERE confidence < ? ORDER BY confidence LIMIT ?`.
**2. In plain words:** These are the model's least certain answers, and the most valuable ones for a human to relabel. They feed the next training set (active learning).
**3. Example to quote:**
> "'It arrived Tuesday' got confidence 0.526, so it shows up first in my low-confidence list for review."

**Seen in:** `low_confidence`

## PII and what not to log  `(concept)`
**1. Definition:** Personally identifiable information, such as names or emails, which should not be logged carelessly, along with secrets.
**2. In plain words:** Raw inputs may contain personal data. Sample, mask or log only lengths and hashes, and never log secrets or API keys.
**3. Example to quote:**
> "I'd mask emails in the input text before logging it, or store only its length."

**Seen in:** `WALKTHROUGH.md` rebuild checklist and "Say it aloud"

## sqlite3.OperationalError  `(standard library)`
**1. Definition:** The exception raised for database-level problems such as a missing table or column.
**2. In plain words:** A typo like `lable` gives `no such column: lable`. Compare your query with the `CREATE TABLE`, or open the DB in the `sqlite3` CLI and run `.schema`.
**3. Example to quote:**
> "Renaming a column in a query to `lable` raised `OperationalError: no such column: lable`, so I checked my schema."

**Seen in:** `BREAK IT` comment, `WALKTHROUGH.md`

## Reusing inference code from step 02  `(concept)`
**1. Definition:** The demo imports `load_artifact`, `predict_text` and `train` from the previous step instead of rewriting them.
**2. In plain words:** It shows how modules compose: the same inference code feeds the logging layer. It also makes sure an artifact exists by training if none is found.
**3. Example to quote:**
> "`from predict import load_artifact, predict_text` reuses step 02's inference pipeline, and the log records its output."

**Seen in:** top of `main.py`

## threshold=0.0 for raw labels  `(concept)`
**1. Definition:** Setting the confidence threshold to 0 means every prediction returns its best label, never "uncertain".
**2. In plain words:** For logging we want the model's actual best guess plus its confidence, and decide later which rows are doubtful.
**3. Example to quote:**
> "I call `predict_text(model, text, threshold=0.0)` so the log always stores the raw best label."

**Seen in:** `main.py`

## Warm-up latency  `(concept)`
**1. Definition:** The first few calls to a model or library are slower because of lazy initialisation and caches.
**2. In plain words:** That is why p95 (11 ms) was far above p50 (0.3 ms). Real services run a few warm-up predictions at startup before taking traffic.
**3. Example to quote:**
> "My max latency of 24 ms is the first call; after warm-up the typical call is 0.3 ms."

**Seen in:** `WALKTHROUGH.md`

## Repeating a list with *  `(syntax)`
**1. Definition:** `list * 5` repeats the list five times.
**2. In plain words:** A cheap way to make 40 requests from 8 sample texts for the demo.
**3. Example to quote:**
> "`traffic = [...8 texts...] * 5` gives 40 predictions to log."

```python
print([1, 2] * 2)   # [1, 2, 1, 2]
```

**Seen in:** `main.py`

## Docstring-level design: why log predictions  `(concept)`
**1. Definition:** Logging serves debugging, monitoring (drift and latency), auditing and creating future training data from real traffic and user feedback.
**2. In plain words:** Without a log you are blind: you cannot reproduce an incident or tell whether the model is getting worse. The cost is write load and privacy risk, handled by sampling and masking.
**3. Example to quote:**
> "I log predictions because they let me debug incidents, spot drift, and build my next training set from real usage."

**Seen in:** `prediction_log.py` docstring
