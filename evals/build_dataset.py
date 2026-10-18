"""Build evals/dataset.jsonl from evals/questions.yaml by executing each gold query.

Usage:  uv run python -m evals.build_dataset
"""

import json
from pathlib import Path

import yaml

from sql_analyst.config import get_settings
from sql_analyst.tools.executor import execute_sql

EVALS = Path(__file__).resolve().parent
SOURCE = EVALS / "questions.yaml"
DATASET = EVALS / "dataset.jsonl"
MALICIOUS = EVALS / "malicious.jsonl"


def load_source() -> dict:
    return yaml.safe_load(SOURCE.read_text())


def build(db_path=None) -> tuple[list[dict], list[dict]]:
    db_path = db_path or get_settings().db_path
    src = load_source()
    examples = []
    for q in src["questions"]:
        result = execute_sql(q["gold_sql"], db_path, max_rows=1000)
        examples.append(
            {
                "id": q["id"],
                "difficulty": q["difficulty"],
                "question": q["question"].strip(),
                "gold_sql": q["gold_sql"].strip(),
                "expected_rows": result.rows,
            }
        )
    malicious = [{"id": m["id"], "question": m["question"].strip()} for m in src["malicious"]]
    return examples, malicious


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


if __name__ == "__main__":
    examples, malicious = build()
    write_jsonl(DATASET, examples)
    write_jsonl(MALICIOUS, malicious)
    counts = {d: sum(e["difficulty"] == d for e in examples) for d in ("easy", "medium", "hard")}
    print(f"Wrote {len(examples)} questions {counts} -> {DATASET}")
    print(f"Wrote {len(malicious)} malicious prompts -> {MALICIOUS}")
