"""The model providers this project supports.

**Chat and embeddings are configured separately, and that is the whole point of
this module's shape.** They are different jobs with different constraints: the
chat model answers, routes and grades, so it is worth picking for quality or
cost; the embedding model indexes the corpus, and changing it later means
rebuilding the index. There is no reason the same vendor should have to do both,
and plenty of reasons not to — Groq is fast and cheap for the routing calls but
serves no embeddings at all, so forcing one provider for both would quietly rule
it out.

So a user supplies two keys. Where they happen to pick the same provider for
both, the UI offers to reuse the one key rather than making them paste it twice.
"""

# Everything that can answer, route and grade. These are the providers whose
# langchain packages this service actually installs.
CHAT_PROVIDERS = {
    "openai": {
        "label": "OpenAI",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
        "key_hint": "Starts with sk-… from platform.openai.com",
    },
    "google": {
        "label": "Google Gemini",
        "models": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
        "key_hint": "From aistudio.google.com",
    },
    "anthropic": {
        "label": "Anthropic",
        "models": ["claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250929"],
        "key_hint": "Starts with sk-ant-… from console.anthropic.com",
    },
    "groq": {
        "label": "GroqCloud",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        "key_hint": "Starts with gsk_… from console.groq.com — fast and cheap for routing",
    },
}

# Only these two serve embeddings here. Deliberately a shorter list: an
# embedding model cannot be swapped casually, because the index is built with it
# and a different model means different dimensions and a full re-index.
EMBED_PROVIDERS = {
    "openai": {
        "label": "OpenAI",
        "models": ["text-embedding-3-small", "text-embedding-3-large"],
        "key_hint": "The same sk-… key works for both chat and embeddings",
    },
    "google": {
        "label": "Google Gemini",
        "models": ["models/text-embedding-004", "gemini-embedding-001"],
        "key_hint": "The same aistudio.google.com key works for both",
    },
}


class ProviderError(Exception):
    pass


def build_llm(provider: str, model: str, api_key: str,
              temperature: float = 0.2, max_tokens: int = 1600):
    """One chat model.

    Temperature is low by default and lower still at the routing and grading
    nodes: those classify rather than write, and a classifier that varies
    between identical inputs cannot be evaluated.
    """
    if not api_key:
        raise ProviderError("No chat API key configured. Add one in Setup first.")
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model, google_api_key=api_key,
            temperature=temperature, max_output_tokens=max_tokens)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, api_key=api_key,
                             temperature=temperature, max_tokens=max_tokens, timeout=90)
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, api_key=api_key,
                        temperature=temperature, max_tokens=max_tokens, timeout=90)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=api_key, temperature=temperature,
                          max_tokens=max_tokens, timeout=90)
    raise ProviderError(f"Unknown chat provider: {provider}")


def build_embeddings(provider: str, model: str, api_key: str):
    """One embedding model, for indexing and for the query side of retrieval.

    Both sides must use the same one. Embedding a query with a different model
    from the corpus produces vectors in an unrelated space, and the search
    silently returns nonsense rather than failing.
    """
    if not api_key:
        raise ProviderError("No embedding API key configured. Add one in Setup first.")
    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model, api_key=api_key)
    raise ProviderError(f"Unknown embedding provider: {provider}")


def catalog() -> dict:
    return {
        "chat": [{"provider": key, **value} for key, value in CHAT_PROVIDERS.items()],
        "embedding": [{"provider": key, **value} for key, value in EMBED_PROVIDERS.items()],
        "note": ("Chat and embeddings are configured separately. They can be the same provider "
                 "or different ones — Groq answers quickly and cheaply but serves no embeddings, "
                 "so pairing it with OpenAI or Google for indexing is a reasonable choice."),
    }


def known_chat(provider: str) -> bool:
    return provider in CHAT_PROVIDERS


def known_embedding(provider: str) -> bool:
    return provider in EMBED_PROVIDERS
