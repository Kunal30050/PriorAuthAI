from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv

try:
    from tracing import log_event
except ImportError:
    from agent.tracing import log_event

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

# Per-1K-token pricing, USD -- fill in your actual Gemini pricing tier here.
# Left at 0 by default so cost_per_successful_task doesn't silently report a
# wrong number; a 0 cost in the dashboard is your cue to set these.
GEMINI_PRICE_PER_1K_INPUT = float(os.getenv("GEMINI_PRICE_PER_1K_INPUT", "0"))
GEMINI_PRICE_PER_1K_OUTPUT = float(os.getenv("GEMINI_PRICE_PER_1K_OUTPUT", "0"))


def _estimate_cost(tokens_in: int | None, tokens_out: int | None) -> float | None:
    if tokens_in is None and tokens_out is None:
        return None
    return (
        (tokens_in or 0) / 1000 * GEMINI_PRICE_PER_1K_INPUT
        + (tokens_out or 0) / 1000 * GEMINI_PRICE_PER_1K_OUTPUT
    )


class LLMConfigError(Exception):
    pass


def get_provider() -> str:
    return (os.getenv("LLM_PROVIDER") or "mock").strip().lower()


def get_model_name() -> str:
    return (os.getenv("LLM_MODEL") or "gemini-3.8-flash-lite").strip()


def get_gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None


def get_openrouter_api_key() -> str | None:
    return os.getenv("OPENROUTER_API_KEY") or None


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


def create_openrouter_chat_model(
    api_key: str | None = None,
    model: str | None = None,
):
    """Initialize a LangChain chat model pointed at OpenRouter's
    OpenAI-compatible endpoint. Does not send a request.

    OpenRouter free-tier model IDs end in ':free' (e.g.
    'meta-llama/llama-3.3-70b-instruct:free'), or use 'openrouter/free' to
    let OpenRouter route to whichever free model is currently available --
    useful since the free lineup rotates without notice.

    provider.sort="latency" tells OpenRouter to always route to whichever
    backing provider is currently fastest for this model, instead of its
    default price-based load balancing. This directly targets P95 tail
    latency (the "unlucky, landed on a slow/queued provider" case) at the
    cost of possibly paying a bit more on a non-free model. It never
    changes what free-tier request limits apply.
    """
    from langchain_openai import ChatOpenAI

    key = api_key if api_key is not None else get_openrouter_api_key()
    if not key:
        raise LLMConfigError(
            "LLM_PROVIDER=openrouter requires OPENROUTER_API_KEY."
        )
    model_name = model or get_model_name()
    return ChatOpenAI(
        model=model_name,
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        timeout=45,
        extra_body={"provider": {"sort": "latency"}},
    )


def invoke_llm_traced(
    llm,
    prompt: str,
    conversation_id: str,
    workflow_id: str,
    step_name: str,
    prompt_version: str = "v1",
):
    """
    Wraps llm.invoke(prompt) with tracing. Use this in workflow_graph.py
    everywhere you currently call llm.invoke(prompt) directly -- it returns
    the exact same response object, so nothing downstream needs to change.

    Pulls token counts from response.usage_metadata, which
    langchain_google_genai populates for Gemini calls -- if a future
    provider/version doesn't set it, tokens_in/out just log as None rather
    than raising.
    """
    start = time.time()
    error = None
    response = None
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        usage = getattr(response, "usage_metadata", None) if response else None
        tokens_in = usage.get("input_tokens") if usage else None
        tokens_out = usage.get("output_tokens") if usage else None
        log_event(
            conversation_id=conversation_id or "untagged",
            workflow_id=workflow_id or "untagged",
            step_name=step_name,
            event_type="llm_call",
            model_name=get_model_name(),
            prompt_version=prompt_version,
            latency_ms=(time.time() - start) * 1000,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=_estimate_cost(tokens_in, tokens_out),
            error=error,
        )


def get_chat_model():
    """Return a LangChain chat model, or None in mock mode."""
    provider = get_provider()
    if provider in ("mock", "none", ""):
        return None
    if provider == "gemini":
        return create_gemini_chat_model()
    if provider == "openrouter":
        return create_openrouter_chat_model()
    raise LLMConfigError(
        f"Unsupported LLM_PROVIDER '{provider}'. Use 'mock', 'gemini', or 'openrouter'."
    )