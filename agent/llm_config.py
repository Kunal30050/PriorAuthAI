from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


class LLMConfigError(Exception):
    pass


def get_provider() -> str:
    return (os.getenv("LLM_PROVIDER") or "mock").strip().lower()


def get_model_name() -> str:
    return (os.getenv("LLM_MODEL") or "gemini-2.0-flash").strip()


def get_gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None


def is_mock() -> bool:
    return get_provider() in ("mock", "none", "")


def create_gemini_chat_model(
    api_key: str | None = None,
    model: str | None = None,
):
    """Initialize LangChain's Gemini chat model. Does not send a request."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    key = api_key if api_key is not None else get_gemini_api_key()
    if not key:
        raise LLMConfigError(
            "LLM_PROVIDER=gemini requires GEMINI_API_KEY (or GOOGLE_API_KEY)."
        )
    model_name = model or get_model_name()
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=key,
        temperature=0,
    )


def get_chat_model():
    """Return a LangChain chat model, or None in mock mode."""
    provider = get_provider()
    if provider in ("mock", "none", ""):
        return None
    if provider == "gemini":
        return create_gemini_chat_model()
    raise LLMConfigError(
        f"Unsupported LLM_PROVIDER '{provider}'. Use 'mock' or 'gemini'."
    )
