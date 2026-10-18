"""Run the evaluation for one configuration.

Usage:
    uv run python -m evals.run_eval --config fewshot
    uv run python -m evals.run_eval --config zeroshot --delay 5
    uv run python -m evals.run_eval --config fewshot --limit 3 --fresh   # quick smoke run
    uv run python -m evals.run_eval --config fewshot --langsmith          # also log to LangSmith

Results are appended one record at a time to evals/results/<config>/, so an
interrupted run (rate limits, laptop sleep) resumes where it stopped.
"""

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from evals import core
from evals.build_dataset import DATASET, MALICIOUS
from evals.configs import CONFIGS
from sql_analyst.config import get_settings
from sql_analyst.graph.service import create_service

RESULTS = Path(__file__).resolve().parent / "results"
log = logging.getLogger("evals")


def make_service(config: str):
    settings = get_settings().model_copy(update=CONFIGS[config])
    # Fresh in-memory checkpoints: eval threads shouldn't pollute the app's conversation DB.
    return create_service(settings, checkpointer=InMemorySaver()), settings


def run(config: str, delay: float | None, limit: int | None, fresh: bool, malicious: bool) -> dict:
    out_dir = RESULTS / config
    if fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    records_path, malicious_path = out_dir / "records.jsonl", out_dir / "malicious.jsonl"

    service, settings = make_service(config)
    delay = settings.eval_delay_s if delay is None else delay
    examples = core.load_jsonl(DATASET)[:limit]
    done = {r["id"] for r in core.load_jsonl(records_path)}
    todo = [e for e in examples if e["id"] not in done]
    log.info(
        "config=%s llm=%s: %d done, %d to run",
        config,
        service.metadata["llm"],
        len(done),
        len(todo),
    )

    for i, example in enumerate(todo, start=1):
        record = core.run_example(service, example)
        core.append_jsonl(records_path, record)
        mark = "✓" if record["strict_match"] else ("~" if record["lenient_match"] else "✗")
        log.info(
            "[%d/%d] %s %s %s retries=%d %.1fs",
            i, len(todo), mark, record["id"], record["status"], record["retries"],
            record["latency_ms"] / 1000,
        )  # fmt: skip
        time.sleep(delay)  # stay under Groq free-tier tokens/minute

    db_unchanged = None
    if malicious:
        before = core.file_sha256(settings.db_path)
        done_mal = {r["id"] for r in core.load_jsonl(malicious_path)}
        for example in [m for m in core.load_jsonl(MALICIOUS) if m["id"] not in done_mal]:
            record = core.run_malicious(service, example)
            core.append_jsonl(malicious_path, record)
            log.info(
                "malicious %s -> %s (%s)", record["id"], record["status"], record["blocked_by"]
            )
            time.sleep(delay)
        db_unchanged = core.file_sha256(settings.db_path) == before

    ids = {e["id"] for e in examples}
    records = [r for r in core.load_jsonl(records_path) if r["id"] in ids]
    summary = {
        "config": config,
        "settings": {**CONFIGS[config], "max_sql_retries": settings.max_sql_retries},
        "llm": service.metadata["llm"],
        **core.summarize(records, core.load_jsonl(malicious_path), db_unchanged),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="seconds between questions (default: EVAL_DELAY_S)",
    )
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    parser.add_argument(
        "--fresh", action="store_true", help="discard previous results for this config"
    )
    parser.add_argument("--no-malicious", action="store_true", help="skip the safety prompts")
    parser.add_argument(
        "--langsmith", action="store_true", help="run as a LangSmith experiment instead"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.langsmith:
        from evals.langsmith_eval import run_langsmith_experiment

        run_langsmith_experiment(args.config, delay=args.delay)
        return

    summary = run(args.config, args.delay, args.limit, args.fresh, not args.no_malicious)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_difficulty"}, indent=2))


if __name__ == "__main__":
    main()
