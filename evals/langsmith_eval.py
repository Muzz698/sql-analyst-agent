"""LangSmith integration: upload the dataset and run experiments.

    uv run python -m evals.langsmith_eval upload            # create/refresh the dataset
    uv run python -m evals.run_eval --config fewshot --langsmith

Requires LANGSMITH_API_KEY (free tier). The evaluators reuse evals.core so local
and LangSmith scores are computed identically.
"""

import sys
import time

from evals import core
from evals.build_dataset import DATASET

DATASET_NAME = "sql-analyst-chinook-40"


def _require_tracing():
    from sql_analyst.config import get_settings
    from sql_analyst.tracing import configure_tracing

    if not configure_tracing(get_settings().model_copy(update={"langsmith_tracing": True})):
        sys.exit("LANGSMITH_API_KEY is not set; add it to .env to use LangSmith evals.")


def upload_dataset() -> str:
    from langsmith import Client

    _require_tracing()
    client = Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        client.delete_dataset(dataset_name=DATASET_NAME)
    ds = client.create_dataset(DATASET_NAME, description="40 Chinook questions with gold SQL")
    client.create_examples(
        dataset_id=ds.id,
        examples=[
            {
                "inputs": {"question": e["question"]},
                "outputs": {"gold_sql": e["gold_sql"], "expected_rows": e["expected_rows"]},
                "metadata": {"id": e["id"], "difficulty": e["difficulty"]},
            }
            for e in core.load_jsonl(DATASET)
        ],
    )
    return DATASET_NAME


# --- Evaluators (outputs = target() result, reference_outputs = dataset outputs) ----


def strict_match(outputs: dict, reference_outputs: dict) -> dict:
    ok = outputs["valid_sql"] and core.results_match(
        reference_outputs["expected_rows"], outputs["rows"]
    )
    return {"key": "execution_accuracy_strict", "score": int(ok)}


def lenient_match(outputs: dict, reference_outputs: dict) -> dict:
    ok = outputs["valid_sql"] and core.results_match(
        reference_outputs["expected_rows"], outputs["rows"], allow_extra_columns=True
    )
    return {"key": "execution_accuracy_lenient", "score": int(ok)}


def sql_valid(outputs: dict) -> dict:
    return {"key": "sql_valid", "score": int(outputs["valid_sql"])}


def retries(outputs: dict) -> dict:
    return {"key": "retries", "score": outputs["retries"]}


def make_target(service, delay: float):
    def target(inputs: dict) -> dict:
        result, approvals = core.ask_with_auto_approval(
            service, f"ls-{time.time_ns()}", inputs["question"]
        )
        time.sleep(delay)
        return {
            "sql": result.sql,
            "rows": result.rows,
            "answer": result.answer,
            "status": result.status,
            "retries": result.retries,
            "approvals": approvals,
            "valid_sql": result.status == "answered" and result.sql is not None,
        }

    return target


def run_langsmith_experiment(config: str, delay: float | None = None):
    from langsmith import evaluate

    from evals.configs import CONFIGS
    from evals.run_eval import make_service

    _require_tracing()
    service, settings = make_service(config)
    return evaluate(
        make_target(service, settings.eval_delay_s if delay is None else delay),
        data=DATASET_NAME,
        evaluators=[strict_match, lenient_match, sql_valid, retries],
        experiment_prefix=config,
        metadata={"config": config, **CONFIGS[config]},
        max_concurrency=1,  # sequential: free-tier rate limits
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["upload"]:
        print("Uploaded dataset:", upload_dataset())
    else:
        sys.exit("usage: python -m evals.langsmith_eval upload")
