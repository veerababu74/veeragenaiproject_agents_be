from typing import Literal

from pydantic import BaseModel, Field

PROVIDERS = ("openai", "google", "openrouter", "groq", "anthropic")

Provider = Literal["openai", "google", "openrouter", "groq", "anthropic"]


class AgentRequest(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    description: str = Field(default="", max_length=400)
    system_prompt: str = Field(default="", max_length=8000)
    provider: Provider = "openai"
    # Free text rather than an enum: providers add and retire model names
    # constantly, and a fixed list would block a new one until redeploy.
    model: str = Field(min_length=1, max_length=120)
    temperature: float = Field(default=0.3, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=256, le=16384)
    api_key: str = Field(default="", max_length=400)


class ProviderKeyRequest(BaseModel):
    provider: Provider
    api_key: str = Field(min_length=8, max_length=400)


class BuiltinToolRequest(BaseModel):
    tool_type: str = Field(min_length=1, max_length=40)
    name: str = Field(default="", max_length=60)
    description: str = Field(default="", max_length=400)
    config: dict = Field(default_factory=dict)


class CustomToolField(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    type: Literal["string", "integer", "number", "boolean"] = "string"
    description: str = Field(default="", max_length=200)
    required: bool = True


class CustomToolRequest(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    description: str = Field(min_length=1, max_length=400)
    api_url: str = Field(min_length=8, max_length=500)
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "GET"
    headers: dict = Field(default_factory=dict)
    auth_type: Literal["none", "bearer", "api_key"] = "none"
    auth_config: dict = Field(default_factory=dict)
    fields: list[CustomToolField] = Field(default_factory=list)


class RunRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str = Field(default="", max_length=80)
