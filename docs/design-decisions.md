# Design decisions & trade-offs

Each section says what was chosen, why, and what it costs.

## 1. Safety: three independent layers

| Layer | Mechanism | Catches |
|---|---|---|
| Router | LLM classifies requests to modify data as out of scope | Plain-English attacks ("delete all customers") |
| Safety guard | sqlglot AST; walks *every* node | DML/DDL anywhere (including inside CTEs), `PRAGMA`, `ATTACH`, multi-statements, `load_extension` |
| Executor | `mode=ro` URI + `PRAGMA query_only` | Anything that slips past both (tested by running `DELETE` directly) |

**Why AST, not regex:** regex is defeated by comments (`DrOp /* x */ TABLE`) and has no notion of
structure. A top-level-only check is also not enough: `WITH d AS (DELETE … RETURNING *) SELECT * FROM d`
parses as a `Select` at the root. This was found while probing sqlglot before writing the guard,
and it has a regression test.

**Unsafe vs invalid:** the guard returns three outcomes. `invalid` (typo, not a query) goes back to the
writer as an error; `unsafe` ends the run. If unsafe SQL were retried, an adversarial prompt
would get four attempts at the guard.

**Cost:** the guard is deliberately conservative. Read-only constructs it doesn't model would be
rejected as `Command`, which is the right default for a public demo.

## 2. Self-correction

Execution errors, syntax errors and **empty results** feed back to the writer with the exact SQLite
error message ("no such column: Nme") plus all prior attempts, so the model doesn't repeat itself.
Empty results are retried because they usually mean a wrong filter value (`'acdc'` vs `'AC/DC'`).
After the retry cap, an empty result is accepted as a legitimate "no matching data" answer.

`retries` is derived as `len(attempts)` rather than kept as a separate counter, so "max 3 retries"
has one source of truth.

## 3. Human-in-the-loop

The graph pauses with `interrupt()` when the guard flags a query (more than 3 joins, a cartesian join,
`SELECT *` over a large table with no filter) or the router's confidence is below 0.6. State is
checkpointed, so the pause survives a server restart (tested). Approval can **edit** the SQL; the
edited SQL goes back through the guard, so a human can't paste `DROP TABLE` past it either (tested).

**Trade-off:** the heuristics are structural. Cost-based flagging (`EXPLAIN QUERY PLAN`) would be
more precise but is SQLite-specific and noisier on a small database.

## 4. LLMs choose, code executes

- **Schema agent:** the LLM picks tables from one-line summaries; code validates names and adds
  bridge tables by BFS over foreign keys (Customer + Genre → Invoice, InvoiceLine, Track).
- **Chart agent:** the LLM returns `{type, x, y}`; code validates the columns exist and `y` is
  numeric, then builds the Plotly figure. A hallucinated column produces no chart rather than a
  broken one.
- **Analyst:** a single-row result never gets a chart, regardless of what the LLM says.

This keeps LLM output small, which matters on a free tier with a token-per-minute limit, and
keeps failure modes predictable.

## 5. Structured output everywhere

Every agent returns a Pydantic model, never free text to parse. Two findings from live runs:

1. **gpt-oss sometimes echoed the JSON schema** (`{"description": …, "properties": …}`) instead of
   filling it in, failing parsing 4 times in a row on one turn. The fix was Groq's **strict** mode,
   which constrains decoding. All five schemas were tested in strict mode before switching. The raw
   output of any parse failure is now logged, which is how this was diagnosed.
2. **The analyst rescaled a number** ("$27.72" became "$27.72k"). The prompt now requires copying
   values exactly, and the case was re-verified against SQLite.

## 6. Provider abstraction & resilience

Agents depend on one method: `generate(schema, messages) → Pydantic object`. That gives:

- Groq / Ollama switching through config, plus a scripted `FakeLLM` for tests, so CI needs no keys.
- One retry policy: exponential backoff with jitter that also honours `Retry-After`. Only transient
  failures are retried (429s, connection errors, malformed JSON); bugs fail immediately.
- **Daily quota errors are not retried.** Groq's free tier caps tokens *per day* (200k for
  gpt-oss-120b). Those 429s can't clear in minutes; retrying wasted 4 minutes per call before this
  was fixed. Now the UI shows a clear message and the eval runner stops cleanly and resumes later.

## 7. Evaluation methodology

- **Result-set comparison, not SQL text:** many different queries are correct.
- **Strict + lenient:** strict-only would score "Iron Maiden, 213" as wrong for "which artist has
  the most tracks?". Lenient accepts it. Reporting both is more honest than picking one.
- **Gold data is generated, not hand-typed:** `build_dataset.py` runs the gold SQL; a test fails if
  `dataset.jsonl` drifts from `questions.yaml`. Top-N questions were checked for ties at the boundary.
- **No leakage:** a test ensures no eval question or gold SQL duplicates a few-shot example.
- **Incomplete runs are compared like-for-like** on the common subset, and untested metrics show
  `n/a`, never `0%`.

## 8. One service, two deployment shapes

The API, the UI and the evals all call `AnalystService.ask / resume / history`. The UI talks to
an `AnalystClient` with two implementations: in-process (one container, for free hosting) and HTTP
(docker-compose). The same image serves both roles.

## 9. Testing strategy

176 tests, all keyless:

- Tools: guard (attack corpus), executor (read-only, timeout), schema introspection.
- Routing: every conditional edge as a pure function.
- Graph end to end with `FakeLLM`: self-correction, retry cap, blocking, approve/edit/reject,
  memory, thread isolation, persistence across restart.
- Tracing structure captured locally with a callback recorder (no LangSmith account needed).
- API via `TestClient`; UI via Streamlit's headless `AppTest` (it caught a real crash on duplicate
  column names).
- Evals: matcher edge cases, dataset integrity, metrics.
