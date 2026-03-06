import os
import logging
import warnings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Suppress langchain_google_genai "Both GOOGLE_API_KEY and GEMINI_API_KEY are set" noise
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai.chat_models").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*GOOGLE_API_KEY.*")
warnings.filterwarnings("ignore", message=".*GEMINI_API_KEY.*")

load_dotenv()

_GOOGLE_DEFAULT = "gemini-3.1-pro-preview"
_ANTHROPIC_DEFAULT = "claude-opus-4-6"


def get_llm(provider: str = None, model: str = None):
    """
    Returns an LLM instance for the requested provider and model.

    Reads LLM_PROVIDER and LLM_MODEL from the environment when the caller
    does not pass explicit values.  Defaults to Google Gemini.

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
        return ChatAnthropic(model=resolved_model, temperature=0, max_retries=2)

    resolved_model = model or _GOOGLE_DEFAULT
    return ChatGoogleGenerativeAI(
        model=resolved_model,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
