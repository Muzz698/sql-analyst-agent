"""Schema agent: pick only the tables a question needs.

Sending the full schema wastes tokens (scarce on Groq's free tier) and gives the
SQL writer more ways to go wrong. The LLM picks tables from one-line summaries;
code then validates the names and adds any bridge tables needed for joins.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from sql_analyst.llm import StructuredLLM
from sql_analyst.schemas import TableInfo, TableSelection
from sql_analyst.tools.schema_introspect import expand_with_join_path, table_summaries

SYSTEM = """You select database tables for a SQL analyst.
Return the minimal set of tables needed to answer the question, using exact table names.
Include tables needed for filters and joins. Bridge tables are added automatically.

Tables:
{summaries}"""


def select_tables(llm: StructuredLLM, question: str, tables: list[TableInfo]) -> list[str]:
    messages = [
        SystemMessage(SYSTEM.format(summaries=table_summaries(tables))),
        HumanMessage(question),
    ]
    selection = llm.generate(TableSelection, messages, run_name="schema_agent")
    chosen = expand_with_join_path(selection.tables, tables)
    # If the model returned nothing usable, fall back to the full schema rather than fail.
    return chosen or [t.name for t in tables]
