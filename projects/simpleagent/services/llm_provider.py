"""The five model providers this project supports, behind one factory.

Model names are suggestions, not a whitelist. Providers publish and retire model
ids constantly, so the UI offers a shortlist and still lets the user type
anything their account can reach.
"""

import uuid
from datetime import datetime

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

PROVIDER_CATALOG = {
    "openai": {
        "label": "OpenAI",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "o4-mini"],
        "key_hint": "Starts with sk-… from platform.openai.com",
    },
    "google": {
        "label": "Google Gemini",
        "models": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"],
        "key_hint": "From aistudio.google.com — the same key can serve embeddings",
    },
    "openrouter": {
        "label": "OpenRouter",
        "models": ["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet", "meta-llama/llama-3.3-70b-instruct",
                   "google/gemini-2.0-flash-001"],
        "key_hint": "Starts with sk-or-… from openrouter.ai",
    },
    "groq": {
        "label": "GroqCloud",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-20b"],
        "key_hint": "Starts with gsk_… from console.groq.com",
    },
    "anthropic": {
        "label": "Anthropic",
        "models": ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001", "claude-opus-4-1-20250805"],
        "key_hint": "Starts with sk-ant-… from console.anthropic.com",
    },
}


class ProviderError(Exception):
    pass


def build_llm(provider: str, model: str, api_key: str, temperature: float = 0.3, max_tokens: int = 2048):
    common = {"temperature": temperature, "max_tokens": max_tokens, "timeout": 120}
    if provider == "openai":
        return ChatOpenAI(model=model, api_key=api_key, **common)
    if provider == "anthropic":
        return ChatAnthropic(model=model, api_key=api_key, **common)
    if provider == "groq":
        return ChatGroq(model=model, api_key=api_key, **common)
    if provider == "google":
        # Gemini names the parameter differently and rejects `timeout`.
        return ChatGoogleGenerativeAI(
            model=model, google_api_key=api_key,
            temperature=temperature, max_output_tokens=max_tokens,
        )
    if provider == "openrouter":
        return ChatOpenAI(
            model=model, api_key=api_key, base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "https://veeragenaiproject-fe.vercel.app",
                             "X-Title": "SimpleAgent"},
            **common,
        )
    raise ProviderError(f"Unsupported provider: {provider}")


def get_key(user_id: str, provider: str) -> str:
    from projects.simpleagent.database import get_db

    conn = get_db()
    row = conn.execute(
        "SELECT api_key FROM provider_keys WHERE user_id=? AND provider=?", (user_id, provider)
    ).fetchone()
    conn.close()
    return row["api_key"] if row else ""


def save_key(user_id: str, provider: str, api_key: str) -> None:
    """One key per provider per user, shared by the agent form and the Keys tab
    so a key entered in either place is the same record — and is removed by the
    same 48-hour sweep."""
    from projects.simpleagent.database import get_db

    api_key = (api_key or "").strip()
    if not api_key:
        return
    now = datetime.utcnow().isoformat()
    conn = get_db()
    conn.execute(
        """INSERT INTO provider_keys (id, user_id, provider, api_key, created_at, updated_at)
           VALUES (?,?,?,?,?,?)
           ON CONFLICT(user_id, provider) DO UPDATE SET api_key=excluded.api_key, updated_at=excluded.updated_at""",
        (str(uuid.uuid4()), user_id, provider, api_key, now, now),
    )
    conn.commit()
    conn.close()


def llm_for_agent(user_id: str, agent: dict):
    api_key = get_key(user_id, agent["provider"])
    if not api_key:
        label = PROVIDER_CATALOG.get(agent["provider"], {}).get("label", agent["provider"])
        raise ProviderError(f"No {label} API key is saved. Add one on the agent or under Keys.")
    return build_llm(agent["provider"], agent["model"], api_key, agent["temperature"], agent["max_tokens"])
