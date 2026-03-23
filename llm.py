import os
import logging
import warnings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# Suppress langchain_google_genai "Both GOOGLE_API_KEY and GEMINI_API_KEY are set" noise
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai.chat_models").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*GOOGLE_API_KEY.*")
warnings.filterwarnings("ignore", message=".*GEMINI_API_KEY.*")
warnings.filterwarnings("ignore", message=".*pydantic.v1.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")

load_dotenv()

# ── LangSmith tracing (opt-in via env var) ────────────────────────────────────
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "ai-hedge-fund")

_GOOGLE_DEFAULT = "gemini-3.1-pro-preview"
_ANTHROPIC_DEFAULT = "claude-opus-4-6"

# Approximate cost per 1M tokens (input / output) — update as pricing changes
_COST_PER_1M: dict[str, dict[str, float]] = {
    "gemini-2.5-flash":           {"input": 0.075, "output": 0.30},
    "gemini-3.1-pro-preview":     {"input": 1.25,  "output": 5.00},
    "claude-opus-4-6":            {"input": 15.0,  "output": 75.0},
    "claude-sonnet-4-6":          {"input": 3.0,   "output": 15.0},
    "claude-haiku-4-5-20251001":  {"input": 0.80,  "output": 4.0},
}


class _TokenUsageCallback(BaseCallbackHandler):
    """Accumulates token usage across all LLM calls in a session."""

    def __init__(self):
        self._input_tokens = 0
        self._output_tokens = 0
        self._calls = 0
        self._model: str | None = None

    def on_llm_end(self, response: LLMResult, **kwargs):
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is not None:
                    meta = getattr(msg, "usage_metadata", None)
                    if meta:
                        self._input_tokens += meta.get("input_tokens", 0)
                        self._output_tokens += meta.get("output_tokens", 0)
                        self._calls += 1

    def summary(self, model: str | None = None) -> dict:
        m = model or self._model
        cost = None
        if m and m in _COST_PER_1M:
            rates = _COST_PER_1M[m]
            cost = (
                self._input_tokens  / 1_000_000 * rates["input"] +
                self._output_tokens / 1_000_000 * rates["output"]
            )
        return {
            "calls":         self._calls,
            "input_tokens":  self._input_tokens,
            "output_tokens": self._output_tokens,
            "total_tokens":  self._input_tokens + self._output_tokens,
            "estimated_cost_usd": cost,
            "model": m,
        }


# Module-level singleton — shared across all agents in a session
_tracker = _TokenUsageCallback()


def get_usage_summary(model: str | None = None) -> dict:
    """Returns aggregated token usage for the current session."""
    return _tracker.summary(model)


def get_llm(provider: str | None = None, model: str | None = None):
    """
    Returns an LLM instance for the requested provider and model.

    Reads LLM_PROVIDER and LLM_MODEL from the environment when the caller
    does not pass explicit values.  Defaults to Google Gemini.

    Attaches a token-usage callback to every returned LLM so that
    get_usage_summary() reflects the full session cost.

    Args:
        provider: "google" or "anthropic". Falls back to $LLM_PROVIDER, then "google".
        model: Model name string. Falls back to $LLM_MODEL, then the provider default.

    Returns:
        A LangChain chat model instance (ChatGoogleGenerativeAI or ChatAnthropic).

    Raises:
        ImportError: If langchain-anthropic is not installed when provider is "anthropic".
    """
    provider = provider or os.getenv("LLM_PROVIDER", "google")
    model = model or os.getenv("LLM_MODEL")

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "langchain-anthropic is required for Anthropic support. "
                "Install it with: pip install langchain-anthropic"
            ) from exc
        resolved_model = model or _ANTHROPIC_DEFAULT
        _tracker._model = resolved_model
        return ChatAnthropic(model=resolved_model, temperature=0, max_retries=2,
                             callbacks=[_tracker])

    resolved_model = model or _GOOGLE_DEFAULT
    _tracker._model = resolved_model
    return ChatGoogleGenerativeAI(
        model=resolved_model,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        callbacks=[_tracker],
    )
