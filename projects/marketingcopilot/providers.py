"""The model providers this project supports, and what each one is used for.

Two providers rather than five, because this project needs chat *and* embeddings
from the same key. OpenAI and Google both offer both, so one key from the user
covers the whole pipeline. Adding Anthropic or Groq would mean asking for a
second key purely for embeddings, which is a worse first-run experience than a
shorter list.

Model names are a shortlist, not a whitelist -- providers retire ids constantly,
so the UI suggests and still accepts anything the account can reach.
"""

PROVIDER_CATALOG = {
    "openai": {
        "label": "OpenAI",
        "chat_models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
        "embed_models": ["text-embedding-3-small", "text-embedding-3-large"],
        "key_hint": "Starts with sk-… from platform.openai.com",
    },
    "google": {
        "label": "Google Gemini",
        "chat_models": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
        "embed_models": ["models/text-embedding-004", "gemini-embedding-001"],
        "key_hint": "From aistudio.google.com — the same key serves embeddings",
    },
}


class ProviderError(Exception):
    pass


def build_llm(provider: str, model: str, api_key: str,
              temperature: float = 0.2, max_tokens: int = 1600):
    """One chat model.

    Temperature is low by default and lower still for the routing and grading
    nodes. Those nodes make classifications, not prose, and a classifier that
    varies between identical inputs is a classifier you cannot evaluate.
    """
    if not api_key:
        raise ProviderError("No API key configured. Add one in Setup first.")
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model, google_api_key=api_key,
            temperature=temperature, max_output_tokens=max_tokens)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=api_key, temperature=temperature,
                          max_tokens=max_tokens, timeout=90)
    raise ProviderError(f"Unknown provider: {provider}")


def catalog() -> list[dict]:
    return [{"provider": key, **value} for key, value in PROVIDER_CATALOG.items()]


def known(provider: str) -> bool:
    return provider in PROVIDER_CATALOG
