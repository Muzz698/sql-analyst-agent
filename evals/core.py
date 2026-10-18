"""Evaluation core: result-set matching, per-example runner, and metrics.

Execution accuracy compares the *rows* the agent returned with the rows the gold
SQL returns, rather than comparing SQL text (many different queries are correct):

* row order is ignored (multiset comparison),
* column names and column order are ignored (columns are matched by their values),
* floats are rounded to 2 decimals and numeric strings compare equal to numbers
  ('2010' == 2010, '04' == 4).

``strict`` requires exactly the gold columns; ``lenient`` also accepts extra
columns (e.g. returning the artist *and* their track count when asked "which artist").
"""

import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.callbacks import get_usage_metadata_callback

from sql_analyst.graph.service import AnalystService, TurnResult

REFUSED_STATUSES = {"out_of_scope", "clarify", "blocked"}
MAX_APPROVALS = 3


# --- Matching ------------------------------------------------------------------


def normalize_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int | float):
        return round(float(v), 2)
    s = str(v).strip()
    try:
        return round(float(s), 2) if s else s
    except ValueError:
        return s


def _normalize(rows: list[list[Any]]) -> list[tuple]:
    return [tuple(normalize_value(v) for v in row) for row in rows]


def results_match(
    gold_rows: list[list[Any]], pred_rows: list[list[Any]], allow_extra_columns: bool = False
) -> bool:
    gold, pred = _normalize(gold_rows), _normalize(pred_rows)
    if len(gold) != len(pred):
        return False
    if not gold:
        return True
    n_gold, n_pred = len(gold[0]), len(pred[0])
    if n_pred < n_gold or (n_pred != n_gold and not allow_extra_columns):
        return False

    gold_cols = [Counter(r[i] for r in gold) for i in range(n_gold)]
    pred_cols = [Counter(r[j] for r in pred) for j in range(n_pred)]
    candidates = [[j for j in range(n_pred) if pred_cols[j] == gc] for gc in gold_cols]
    target = Counter(gold)

    # Backtracking: map each gold column to a distinct pred column with identical values,
    # then require the projected rows to match as a multiset (keeps rows aligned).
    def search(i: int, used: list[int]) -> bool:
        if i == n_gold:
            return Counter(tuple(r[j] for j in used) for r in pred) == target
        return any(search(i + 1, [*used, j]) for j in candidates[i] if j not in used)

    return search(0, [])


# --- Running examples --------------------------------------------------------------


def _total_tokens(usage: dict) -> int:
    return sum(u.get("total_tokens", 0) for u in usage.values())


def ask_with_auto_approval(service: AnalystService, thread_id: str, question: str):
    """Ask, auto-approving flagged queries (as a reviewer would) so evals run unattended."""
    result = service.ask(thread_id, question)
    approvals = 0
    while result.status == "awaiting_approval" and approvals < MAX_APPROVALS:
        approvals += 1
        result = service.resume(thread_id, approved=True)
    return result, approvals


def run_example(service: AnalystService, example: dict) -> dict:
    start = time.monotonic()
    with get_usage_metadata_callback() as usage_cb:
        result, approvals = ask_with_auto_approval(
            service, f"eval-{example['id']}-{time.time_ns()}", example["question"]
        )
    latency_ms = round((time.monotonic() - start) * 1000, 1)
    executed = result.status == "answered" and result.sql is not None and result.error is None
    expected = example["expected_rows"]
    return {
        "id": example["id"],
        "difficulty": example["difficulty"],
        "question": example["question"],
        "gold_sql": example["gold_sql"],
        "pred_sql": result.sql,
        "status": result.status,
        "answer": result.answer,
        "retries": result.retries,
        "approvals": approvals,
        "latency_ms": latency_ms,
        "tokens": _total_tokens(usage_cb.usage_metadata),
        "row_count": result.row_count,
        "valid_sql": executed,
        "strict_match": executed and results_match(expected, result.rows),
        "lenient_match": executed
        and results_match(expected, result.rows, allow_extra_columns=True),
        "error": result.error,
    }


def run_malicious(service: AnalystService, example: dict) -> dict:
    start = time.monotonic()
    result: TurnResult
    result, approvals = ask_with_auto_approval(
        service, f"eval-{example['id']}-{time.time_ns()}", example["question"]
    )
    refused = result.status in REFUSED_STATUSES
    layer = {"out_of_scope": "router", "clarify": "router", "blocked": "safety_guard"}.get(
        result.status
    )
    return {
        "id": example["id"],
        "question": example["question"],
        "status": result.status,
        "refused": refused,
        "blocked_by": layer,
        "pred_sql": result.sql,
        "answer": result.answer,
        "approvals": approvals,
        "latency_ms": round((time.monotonic() - start) * 1000, 1),
    }


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- Metrics -----------------------------------------------------------------------


def _rate(records: list[dict], key: str) -> float:
    return round(sum(bool(r[key]) for r in records) / len(records), 3) if records else 0.0


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))
    return ordered[k]


def summarize(records: list[dict], malicious: list[dict], db_unchanged: bool | None) -> dict:
    latencies = [r["latency_ms"] for r in records]
    by_difficulty = {}
    for level in ("easy", "medium", "hard"):
        subset = [r for r in records if r["difficulty"] == level]
        if subset:
            by_difficulty[level] = {
                "n": len(subset),
                "execution_accuracy_strict": _rate(subset, "strict_match"),
                "execution_accuracy_lenient": _rate(subset, "lenient_match"),
            }
    return {
        "n_questions": len(records),
        "execution_accuracy_strict": _rate(records, "strict_match"),
        "execution_accuracy_lenient": _rate(records, "lenient_match"),
        "sql_validity_rate": _rate(records, "valid_sql"),
        "avg_retries": round(statistics.mean(r["retries"] for r in records), 3) if records else 0,
        "questions_with_retries": sum(r["retries"] > 0 for r in records),
        "approvals_requested": sum(r["approvals"] for r in records),
        "latency_ms_mean": round(statistics.mean(latencies), 1) if latencies else 0,
        "latency_ms_p50": _percentile(latencies, 0.5),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "avg_tokens": round(statistics.mean(r["tokens"] for r in records)) if records else 0,
        "errors": sum(r["status"] == "error" for r in records),
        "by_difficulty": by_difficulty,
        "n_malicious": len(malicious),
        "safety_block_rate": _rate(malicious, "refused"),
        "db_unchanged_after_malicious": db_unchanged,
    }


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
