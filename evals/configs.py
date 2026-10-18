"""Named evaluation configurations. Each overrides a few Settings fields."""

from typing import Any

CONFIGS: dict[str, dict[str, Any]] = {
    # Default production setup.
    "fewshot": {
        "llm_provider": "groq",
        "groq_model": "openai/gpt-oss-120b",
        "use_few_shot": True,
        "use_schema_pruning": True,
    },
    # Ablation: identical but without few-shot examples (isolates their effect).
    "zeroshot": {
        "llm_provider": "groq",
        "groq_model": "openai/gpt-oss-120b",
        "use_few_shot": False,
        "use_schema_pruning": True,
    },
    # Smaller model with the production prompt (cost/latency vs accuracy trade-off).
    "qwen_fewshot": {
        "llm_provider": "groq",
        "groq_model": "qwen/qwen3.8-27b",
        "use_few_shot": True,
        "use_schema_pruning": True,
    },
}
