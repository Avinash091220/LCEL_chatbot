"""
llm.py - one place that knows how to build a chat model.

Identical to Session 2. We talk to OpenRouter (an OpenAI-compatible gateway) through
LangChain's ChatOpenAI class, so switching from GPT to Claude to Gemini to Llama is
just a different model-name string - nothing else in the app changes.
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # reads OPENROUTER_API_KEY from a local .env file (ignored by git)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"

# Label shown in the UI -> OpenRouter model id
AVAILABLE_MODELS = {
    "OpenAI · GPT-4o mini": "openai/gpt-4o-mini",
    "Anthropic · Claude Haiku 4.5": "anthropic/claude-haiku-4.5",
    "Google · Gemini 2.5 Flash Lite": "google/gemini-2.5-flash-lite",
    "Meta · Llama 3.3 70B": "meta-llama/llama-3.3-70b-instruct",
}


def get_llm(model: str = DEFAULT_MODEL, temperature: float = 0.3) -> ChatOpenAI:
    """Return a ChatOpenAI pointed at OpenRouter for the given model id."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set (see .env.example).")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
    )
