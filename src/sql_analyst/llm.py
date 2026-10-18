"""LLM provider abstraction.

Agents depend on one tiny interface, ``StructuredLLM.generate(schema, messages)``,
which always returns a validated Pydantic object. That gives us:

* provider switching (Groq / Ollama) through config alone,
* one place for retry-with-backoff on rate limits, and
* a scripted ``FakeLLM`` so tests and CI run with no API keys and no cost.
"""

import logging
import random
from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeVar

import groq
import httpx
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ValidationError
from tenacity import RetryCallState, Retrying, retry_if_exception, stop_after_attempt

from sql_analyst.config import Settings, get_settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMConfigError(RuntimeError):
    pass


class StructuredLLM(Protocol):
    name: str

    def generate(
        self,
        schema: type[T],
        messages: Sequence[BaseMessage],
        *,
        run_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> T: ...


# --- Retry policy ------------------------------------------------------------

_TRANSIENT = (
    groq.RateLimitError,
    groq.APIConnectionError,
    groq.APITimeoutError,
    groq.InternalServerError,
    httpx.TransportError,
)
# The model occasionally emits JSON that fails validation; a fresh sample usually fixes it.
_MALFORMED = (OutputParserException, ValidationError)


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, _TRANSIENT + _MALFORMED):
        return True
    # Groq returns 400 "tool_use_failed"/"json_validate_failed" for malformed structured output.
    return isinstance(exc, groq.BadRequestError) and "failed" in str(exc).lower()


def _retry_after_seconds(exc: BaseException | None) -> float | None:
    response = getattr(exc, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def backoff_wait(initial: float, maximum: float) -> Callable[[RetryCallState], float]:
    """Exponential backoff with jitter that also honours the server's Retry-After."""

    def wait(state: RetryCallState) -> float:
        exp = min(maximum, initial * 2 ** (state.attempt_number - 1))
        delay = exp * (0.5 + random.random() / 2)  # "equal jitter"
        server = _retry_after_seconds(state.outcome.exception() if state.outcome else None)
        return min(maximum, max(delay, server or 0.0))

    return wait


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    log.warning(
        "LLM call failed (%s), retry %d in %.1fs",
        type(exc).__name__,
        state.attempt_number,
        state.next_action.sleep if state.next_action else 0,
    )


# --- Real providers ----------------------------------------------------------


class LangChainLLM:
    """Wraps any LangChain chat model with structured output + backoff."""

    def __init__(
        self,
        chat_model: BaseChatModel,
        name: str,
        settings: Settings,
        method: str = "json_schema",
    ):
        self.chat_model = chat_model
        self.name = name
        self.method = method
        self._settings = settings
        self._cache: dict[type[BaseModel], Any] = {}

    def _structured(self, schema: type[T]):
        if schema not in self._cache:
            self._cache[schema] = self.chat_model.with_structured_output(schema, method=self.method)
        return self._cache[schema]

    def generate(self, schema, messages, *, run_name, metadata=None):
        s = self._settings
        runnable = self._structured(schema).with_config(
            run_name=run_name,
            metadata={"llm": self.name, **(metadata or {})},
        )
        retrying = Retrying(
            stop=stop_after_attempt(s.llm_max_attempts),
            wait=backoff_wait(s.llm_backoff_initial_s, s.llm_backoff_max_s),
            retry=retry_if_exception(is_retryable),
            before_sleep=_log_retry,
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                result = runnable.invoke(list(messages))
                if result is None:  # parser gave up silently
                    raise OutputParserException(f"No {schema.__name__} parsed from model output")
        return result


def _groq(settings: Settings, model: str) -> LangChainLLM:
    from langchain_groq import ChatGroq

    key = settings.groq_api_key
    if key is None or not key.get_secret_value().strip():
        raise LLMConfigError("GROQ_API_KEY is not set. Add it to .env or use LLM_PROVIDER=ollama.")
    kwargs: dict[str, Any] = {}
    if settings.groq_reasoning_effort:
        kwargs["reasoning_effort"] = settings.groq_reasoning_effort
    chat = ChatGroq(
        model=model,
        api_key=key,
        temperature=settings.llm_temperature,
        max_retries=0,  # our backoff policy is the single source of retries
        **kwargs,
    )
    return LangChainLLM(chat, f"groq:{model}", settings)


def _ollama(settings: Settings, model: str) -> LangChainLLM:
    from langchain_ollama import ChatOllama

    chat = ChatOllama(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        reasoning=False,  # qwen3 "thinking" is slow on CPU and not needed with JSON output
    )
    return LangChainLLM(chat, f"ollama:{model}", settings)


# --- Fake (tests / CI) -------------------------------------------------------

Script = dict[str, list[BaseModel | dict | Exception]] | Callable[[type[BaseModel], list], Any]


class FakeLLM:
    """Scripted LLM for tests.

    ``responses`` maps a schema class name (e.g. ``"SQLDraft"``) to a queue of
    outputs returned in order; an ``Exception`` in the queue is raised instead.
    Alternatively pass a callable ``(schema, messages) -> output``.
    Every call is recorded in ``self.calls`` for assertions.
    """

    name = "fake"

    def __init__(self, responses: Script | None = None):
        self._responses = responses if responses is not None else {}
        self.calls: list[dict[str, Any]] = []

    def generate(self, schema, messages, *, run_name, metadata=None):
        self.calls.append({"schema": schema.__name__, "run_name": run_name, "messages": messages})
        if callable(self._responses):
            out = self._responses(schema, list(messages))
        else:
            queue = self._responses.get(schema.__name__)
            if not queue:
                raise AssertionError(f"FakeLLM has no scripted response for {schema.__name__}")
            out = queue.pop(0)
        if isinstance(out, Exception):
            raise out
        return out if isinstance(out, schema) else schema.model_validate(out)

    def calls_for(self, schema_name: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["schema"] == schema_name]


# --- Factory -----------------------------------------------------------------


def get_llm(
    settings: Settings | None = None, provider: str | None = None, model: str | None = None
) -> StructuredLLM:
    settings = settings or get_settings()
    provider = provider or settings.llm_provider
    if provider == "groq":
        return _groq(settings, model or settings.groq_model)
    if provider == "ollama":
        return _ollama(settings, model or settings.ollama_model)
    if provider == "fake":
        return FakeLLM()
    raise LLMConfigError(f"Unknown LLM provider: {provider}")
