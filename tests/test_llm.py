import groq
import httpx
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from tenacity import RetryCallState

from sql_analyst.config import Settings
from sql_analyst.llm import (
    FakeLLM,
    LangChainLLM,
    LLMConfigError,
    backoff_wait,
    get_llm,
    is_retryable,
)
from sql_analyst.schemas import SQLDraft

_REQ = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def rate_limit(retry_after: str | None = None) -> groq.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    return groq.RateLimitError(
        "rate limited", response=httpx.Response(429, headers=headers, request=_REQ), body=None
    )


def bad_request(msg: str) -> groq.BadRequestError:
    return groq.BadRequestError(msg, response=httpx.Response(400, request=_REQ), body=None)


# --- Fake LLM ----------------------------------------------------------------


def test_fake_llm_returns_scripted_responses_in_order():
    llm = FakeLLM(
        {
            "SQLDraft": [
                {"reasoning": "r", "sql": "SELECT 1"},
                SQLDraft(reasoning="r", sql="SELECT 2"),
            ]
        }
    )
    assert llm.generate(SQLDraft, [], run_name="a").sql == "SELECT 1"
    assert llm.generate(SQLDraft, [], run_name="b").sql == "SELECT 2"
    assert [c["run_name"] for c in llm.calls] == ["a", "b"]


def test_fake_llm_raises_scripted_exception_and_missing_script():
    llm = FakeLLM({"SQLDraft": [ValueError("boom")]})
    with pytest.raises(ValueError):
        llm.generate(SQLDraft, [], run_name="x")
    with pytest.raises(AssertionError, match="no scripted response"):
        llm.generate(SQLDraft, [], run_name="x")


# --- Factory -----------------------------------------------------------------


def test_get_llm_fake_by_default_in_tests():
    assert isinstance(get_llm(), FakeLLM)


def test_groq_without_key_fails_fast():
    with pytest.raises(LLMConfigError, match="GROQ_API_KEY"):
        get_llm(Settings(llm_provider="groq", groq_api_key=None))


def test_groq_and_ollama_construct_without_network():
    s = Settings(groq_api_key="gsk_test")
    assert get_llm(s, provider="groq").name == "groq:openai/gpt-oss-120b"
    assert get_llm(s, provider="ollama").name == "ollama:qwen3:4b"
    assert get_llm(s, provider="groq", model="qwen/qwen3.8-27b").name == "groq:qwen/qwen3.8-27b"


# --- Retry policy ------------------------------------------------------------


@pytest.mark.parametrize(
    "exc,expected",
    [
        (rate_limit(), True),
        (OutputParserException("bad json"), True),
        (bad_request("tool_use_failed: invalid JSON"), True),
        (bad_request("invalid model"), False),
        (ValueError("bug"), False),
    ],
)
def test_is_retryable(exc, expected):
    assert is_retryable(exc) is expected


def _state(attempt: int, exc: BaseException) -> RetryCallState:
    state = RetryCallState(retry_object=None, fn=None, args=(), kwargs={})
    state.attempt_number = attempt
    state.set_exception((type(exc), exc, None))
    return state


def test_backoff_grows_exponentially_and_is_capped():
    wait = backoff_wait(initial=2, maximum=10)
    exc = rate_limit()
    # equal jitter: delay in [exp/2, exp]
    assert 1 <= wait(_state(1, exc)) <= 2
    assert 4 <= wait(_state(3, exc)) <= 8
    assert wait(_state(10, exc)) <= 10


def test_backoff_honours_retry_after_header():
    wait = backoff_wait(initial=1, maximum=60)
    assert wait(_state(1, rate_limit(retry_after="7"))) == 7


class _StubChat:
    """Minimal stand-in for a LangChain chat model."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def with_structured_output(self, schema, method, include_raw=False, strict=None):
        def run(_messages):
            self.calls += 1
            out = self.outcomes.pop(0)
            if isinstance(out, Exception):
                raise out
            return out

        return RunnableLambda(run)


def _fast_settings(**kw) -> Settings:
    return Settings(llm_backoff_initial_s=0.001, llm_backoff_max_s=0.001, **kw)


def test_langchain_llm_retries_rate_limit_then_succeeds():
    chat = _StubChat([rate_limit(), rate_limit(), SQLDraft(reasoning="r", sql="SELECT 1")])
    llm = LangChainLLM(chat, "stub", _fast_settings(llm_max_attempts=5))
    assert llm.generate(SQLDraft, [HumanMessage("q")], run_name="t").sql == "SELECT 1"
    assert chat.calls == 3


def test_langchain_llm_gives_up_after_max_attempts():
    chat = _StubChat([rate_limit()] * 3)
    llm = LangChainLLM(chat, "stub", _fast_settings(llm_max_attempts=3))
    with pytest.raises(groq.RateLimitError):
        llm.generate(SQLDraft, [HumanMessage("q")], run_name="t")
    assert chat.calls == 3


def test_langchain_llm_does_not_retry_non_transient_errors():
    chat = _StubChat([ValueError("bug"), SQLDraft(reasoning="r", sql="SELECT 1")])
    llm = LangChainLLM(chat, "stub", _fast_settings())
    with pytest.raises(ValueError):
        llm.generate(SQLDraft, [HumanMessage("q")], run_name="t")
    assert chat.calls == 1


def test_langchain_llm_retries_when_parser_returns_none():
    chat = _StubChat([None, SQLDraft(reasoning="r", sql="SELECT 1")])
    llm = LangChainLLM(chat, "stub", _fast_settings())
    assert llm.generate(SQLDraft, [HumanMessage("q")], run_name="t").sql == "SELECT 1"


def test_parse_failure_with_raw_output_is_retried():
    """include_raw=True returns {'parsed', 'parsing_error', 'raw'}; a failed parse must retry."""
    from langchain_core.messages import AIMessage

    bad = {"parsed": None, "parsing_error": ValueError("schema echoed"), "raw": AIMessage("{}")}
    good = {"parsed": SQLDraft(reasoning="r", sql="SELECT 1"), "parsing_error": None, "raw": None}
    chat = _StubChat([bad, good])
    llm = LangChainLLM(chat, "stub", _fast_settings())
    assert llm.generate(SQLDraft, [HumanMessage("q")], run_name="t").sql == "SELECT 1"
    assert chat.calls == 2


def test_groq_uses_strict_structured_output():
    assert get_llm(Settings(groq_api_key="gsk_test"), provider="groq").strict is True
    assert get_llm(Settings(), provider="ollama").strict is None
