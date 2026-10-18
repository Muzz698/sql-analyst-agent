import json
import re

import pytest
from evals import core, report
from evals.build_dataset import DATASET, MALICIOUS, build
from evals.langsmith_eval import lenient_match, sql_valid, strict_match

from conftest import turn_script
from sql_analyst.agents.sql_writer import load_few_shot
from sql_analyst.tools.safety_guard import validate_sql

# --- result matching ------------------------------------------------------------


def test_ignores_row_order_and_column_names():
    assert core.results_match([["a", 1], ["b", 2]], [["b", 2], ["a", 1]])


def test_ignores_column_order():
    assert core.results_match([["Rock", 826.65]], [[826.65, "Rock"]])


def test_rounds_floats_to_two_decimals():
    assert core.results_match([[5.65]], [[5.651904761904762]])
    assert not core.results_match([[5.65]], [[5.66]])


def test_numeric_strings_equal_numbers():
    assert core.results_match([["2010"]], [[2010]])
    assert core.results_match([[4]], [["04"]])


def test_extra_columns_strict_vs_lenient():
    gold, pred = [["Iron Maiden"]], [["Iron Maiden", 213]]
    assert not core.results_match(gold, pred)
    assert core.results_match(gold, pred, allow_extra_columns=True)


def test_missing_column_never_matches():
    assert not core.results_match([["Helena", "Holý"]], [["Helena"]], allow_extra_columns=True)


def test_row_count_must_match():
    assert not core.results_match([[1], [2]], [[1]])
    assert not core.results_match([[1]], [[1], [1]])


def test_rows_must_stay_aligned_across_columns():
    """Each column's values match, but pairs are scrambled -> not the same result."""
    gold = [["a", 1], ["b", 2]]
    assert not core.results_match(gold, [["a", 2], ["b", 1]])


def test_duplicate_valued_columns_resolve_by_backtracking():
    gold = [[1, 1, "x"], [1, 2, "y"]]
    assert core.results_match(gold, [["y", 1, 2], ["x", 1, 1]])


def test_empty_results():
    assert core.results_match([], [])
    assert not core.results_match([], [[1]])


def test_nulls_and_booleans():
    assert core.results_match([[None, 1]], [[True, None]])


# --- dataset integrity ------------------------------------------------------------


def _load(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _norm_q(q: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()


def test_dataset_shape():
    examples = _load(DATASET)
    assert len(examples) == 40
    counts = {d: sum(e["difficulty"] == d for e in examples) for d in ("easy", "medium", "hard")}
    assert counts == {"easy": 14, "medium": 14, "hard": 12}
    assert len({e["id"] for e in examples}) == 40
    assert len(_load(MALICIOUS)) == 5


def test_committed_dataset_matches_gold_sql(chinook_db):
    """dataset.jsonl must be regenerated whenever questions.yaml changes."""
    examples, malicious = build(chinook_db)
    assert examples == _load(DATASET)
    assert malicious == _load(MALICIOUS)


def test_gold_sql_passes_safety_guard():
    for e in _load(DATASET):
        assert validate_sql(e["gold_sql"]).status == "ok", e["id"]


def test_no_overlap_with_few_shot_examples():
    """Few-shot examples must not leak eval answers."""
    shots = load_few_shot()
    shot_qs = {_norm_q(s["question"]) for s in shots}
    shot_sql = {" ".join(s["sql"].split()).lower() for s in shots}
    for e in _load(DATASET):
        assert _norm_q(e["question"]) not in shot_qs, e["id"]
        assert " ".join(e["gold_sql"].split()).lower() not in shot_sql, e["id"]


# --- runner (fake LLM) -------------------------------------------------------------


EXAMPLE = {
    "id": "t01",
    "difficulty": "easy",
    "question": "How many artists?",
    "gold_sql": "SELECT COUNT(*) FROM Artist",
    "expected_rows": [[3]],
}


def test_run_example_scores_a_correct_answer(fake_service):
    record = core.run_example(fake_service(turn_script()), EXAMPLE)
    assert record["status"] == "answered"
    assert record["valid_sql"] and record["strict_match"] and record["lenient_match"]
    assert record["retries"] == 0 and record["approvals"] == 0


def test_run_example_scores_a_wrong_answer(fake_service):
    script = turn_script(sql="SELECT COUNT(*) FROM Album")  # 2, not 3
    record = core.run_example(fake_service(script), EXAMPLE)
    assert record["valid_sql"]
    assert not record["strict_match"] and not record["lenient_match"]


def test_run_example_auto_approves_flagged_queries(fake_service):
    script = turn_script(sql="SELECT COUNT(*) FROM Artist a, Album b")  # cartesian -> flagged
    record = core.run_example(fake_service(script), {**EXAMPLE, "expected_rows": [[6]]})
    assert record["approvals"] == 1
    assert record["strict_match"]


def test_run_malicious_records_refusal_layer(fake_service):
    router_refusal = {
        "RouterDecision": [
            {
                "reasoning": "r",
                "intent": "out_of_scope",
                "standalone_question": "q",
                "confidence": 1,
            }
        ]
    }
    rec = core.run_malicious(fake_service(router_refusal), {"id": "x", "question": "delete it"})
    assert rec["refused"] and rec["blocked_by"] == "router"

    guard_block = {**turn_script(sql="DELETE FROM Artist")}
    rec = core.run_malicious(fake_service(guard_block), {"id": "x", "question": "delete it"})
    assert rec["refused"] and rec["blocked_by"] == "safety_guard"


# --- metrics + report ----------------------------------------------------------------


def _record(i, difficulty, strict, lenient, retries=0, latency=1000.0, status="answered"):
    return {
        "id": f"q{i}", "difficulty": difficulty, "question": f"question {i}",
        "strict_match": strict, "lenient_match": lenient, "valid_sql": status == "answered",
        "retries": retries, "approvals": 0, "latency_ms": latency, "tokens": 100, "status": status,
    }  # fmt: skip


def test_summarize_metrics():
    records = [
        _record(1, "easy", True, True),
        _record(2, "easy", False, True, retries=2, latency=3000),
        _record(3, "hard", False, False, retries=1, status="failed"),
        _record(4, "medium", True, True),
    ]
    malicious = [{"refused": True}, {"refused": True}, {"refused": False}, {"refused": True}]
    s = core.summarize(records, malicious, db_unchanged=True)
    assert s["execution_accuracy_strict"] == 0.5
    assert s["execution_accuracy_lenient"] == 0.75
    assert s["sql_validity_rate"] == 0.75
    assert s["avg_retries"] == 0.75
    assert s["questions_with_retries"] == 2
    assert s["latency_ms_p50"] == 1000.0
    assert s["by_difficulty"]["easy"]["execution_accuracy_lenient"] == 1.0
    assert s["safety_block_rate"] == 0.75
    assert s["db_unchanged_after_malicious"] is True


def test_report_builds_comparison_table(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    for name, acc in [("a", 0.9), ("b", 0.7)]:
        (tmp_path / name).mkdir()
        summary = core.summarize([_record(1, "easy", acc > 0.8, True)], [], None)
        summary.update(
            config=name, llm="fake", settings={"use_few_shot": True, "use_schema_pruning": True}
        )
        (tmp_path / name / "summary.json").write_text(json.dumps(summary))
        core.append_jsonl(tmp_path / name / "records.jsonl", _record(1, "easy", acc > 0.8, True))
    text = report.build()
    assert "| Execution accuracy (strict) | 100.0% | 0.0% |" in text
    assert "`b`: questions not matched strictly" in text


@pytest.mark.parametrize("valid,expected", [(True, 1), (False, 0)])
def test_langsmith_evaluators(valid, expected):
    outputs = {"valid_sql": valid, "rows": [[3]], "retries": 0}
    ref = {"expected_rows": [[3]]}
    assert strict_match(outputs, ref)["score"] == expected
    assert lenient_match(outputs, ref)["score"] == expected
    assert sql_valid(outputs)["score"] == expected


def test_safety_rate_is_none_when_malicious_not_run():
    s = core.summarize([_record(1, "easy", True, True)], [], None)
    assert s["safety_block_rate"] is None
    assert report._fmt(s["safety_block_rate"], "pct") == "n/a"
