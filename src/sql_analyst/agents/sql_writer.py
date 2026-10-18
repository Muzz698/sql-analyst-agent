"""SQL writer agent: turn a question + pruned schema into SQLite SQL.

On retries it receives every previous attempt and its error, so it can correct
itself instead of repeating the same mistake.
"""

from functools import lru_cache
from importlib import resources

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from sql_analyst.llm import StructuredLLM
from sql_analyst.schemas import Attempt, SQLDraft

SYSTEM = """You are an expert SQLite analyst. Write ONE read-only SELECT query that answers
the question using only the tables and columns in the schema below.

Rules:
- SQLite dialect. Dates are TEXT 'YYYY-MM-DD HH:MM:SS'; use strftime('%Y', col) etc.
- Use explicit JOIN ... ON with the foreign keys shown. Qualify ambiguous columns.
- Give computed columns short snake_case aliases. ROUND money to 2 decimals.
- Return only the columns needed to answer; add ORDER BY for rankings and time series.
- Use LIMIT only when the question asks for top/bottom N.
- Never modify data. Never use PRAGMA, ATTACH, or multiple statements.

Schema:
{schema}"""

EXAMPLES_HEADER = "\n\nExamples of good queries for this database:\n"


@lru_cache
def load_few_shot() -> list[dict[str, str]]:
    text = resources.files("sql_analyst.prompts").joinpath("few_shot.yaml").read_text()
    return yaml.safe_load(text)


def format_examples(examples: list[dict[str, str]]) -> str:
    return "\n".join(f"Q: {ex['question']}\nSQL:\n{ex['sql'].strip()}\n" for ex in examples)


def format_attempts(attempts: list[Attempt]) -> str:
    parts = [f"Attempt {i}:\n{a.sql}\nProblem: {a.error}" for i, a in enumerate(attempts, start=1)]
    return (
        "\n\nYour previous attempts failed. Fix the problem; do not repeat a failed query.\n\n"
        + "\n\n".join(parts)
    )


def write_sql(
    llm: StructuredLLM,
    question: str,
    schema_text: str,
    few_shot: bool = True,
    attempts: list[Attempt] | None = None,
) -> SQLDraft:
    system = SYSTEM.format(schema=schema_text)
    if few_shot:
        system += EXAMPLES_HEADER + format_examples(load_few_shot())
    human = f"Question: {question}"
    if attempts:
        human += format_attempts(attempts)
    return llm.generate(
        SQLDraft,
        [SystemMessage(system), HumanMessage(human)],
        run_name="sql_writer",
        metadata={"attempt": len(attempts or []) + 1, "few_shot": few_shot},
    )
