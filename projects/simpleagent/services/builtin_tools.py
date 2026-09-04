"""Built-in tools.

Every tool is a plain REST call made with aiohttp rather than a provider SDK, so
adding one never adds a dependency. Each factory returns a StructuredTool, or
None when a required credential is missing, and every network call returns its
failure as text the agent can reason about instead of raising through the graph
— a tool that throws would end the run, when the useful behaviour is for the
agent to notice the failure and try something else.

Half the catalogue needs no API key at all, which is deliberate: the point of
this project is watching an agent pick and sequence tools, and that should not
be gated behind signing up for a search provider.
"""

import ast
import json
import operator
import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote

import aiohttp
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

TIMEOUT = aiohttp.ClientTimeout(total=30)


async def _request(method, url, **kwargs):
    try:
        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            async with session.request(method, url, **kwargs) as response:
                body = await response.text()
                if response.status >= 400:
                    return None, f"HTTP {response.status}: {body[:400]}"
                try:
                    return json.loads(body), None
                except ValueError:
                    return body, None
    except Exception as error:
        return None, f"Request failed: {error}"


# ── date and time ───────────────────────────────────────────────────────────

class DatetimeInput(BaseModel):
    timezone_offset_hours: Optional[float] = Field(
        default=None, description="Offset from UTC in hours, e.g. 5.5 for IST. Omit for the configured default.")
    days_offset: int = Field(default=0, description="Shift by days: 1 for tomorrow, -1 for yesterday.")


def create_datetime_tool(config):
    default_offset = float(config.get("timezone_offset_hours") or 0)

    async def _run(timezone_offset_hours: float = None, days_offset: int = 0):
        offset = default_offset if timezone_offset_hours is None else timezone_offset_hours
        moment = datetime.now(timezone(timedelta(hours=offset))) + timedelta(days=days_offset)
        return json.dumps({
            "date": moment.strftime("%Y-%m-%d"), "time": moment.strftime("%H:%M:%S"),
            "day_of_week": moment.strftime("%A"), "iso": moment.isoformat(), "utc_offset_hours": offset,
        })

    return StructuredTool.from_function(
        coroutine=_run, name="current_datetime", args_schema=DatetimeInput,
        description="Get the current date, time and day of week. Use whenever the answer depends on what today is.")


# ── calculator ──────────────────────────────────────────────────────────────

_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _evaluate(node):
    """Only numeric literals and the operators above are reachable, so no name
    lookup, attribute access or call can be expressed in the input."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("only numbers are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("unsupported expression")


class CalculatorInput(BaseModel):
    expression: str = Field(description='Arithmetic expression, e.g. "(1200 * 0.18) + 45"')


def create_calculator_tool(_config):
    async def _run(expression: str):
        try:
            if len(expression) > 200:
                return "That expression is too long."
            return str(_evaluate(ast.parse(expression, mode="eval").body))
        except Exception as error:
            return f'Could not evaluate "{expression}": {error}'

    return StructuredTool.from_function(
        coroutine=_run, name="calculator", args_schema=CalculatorInput,
        description="Evaluate an arithmetic expression exactly. Use for any calculation instead of doing mental maths.")


# ── web ─────────────────────────────────────────────────────────────────────

class SearchInput(BaseModel):
    query: str = Field(description="The search query")


def create_duckduckgo_tool(config):
    max_results = int(config.get("max_results") or 5)

    async def _run(query: str):
        try:
            from ddgs import DDGS
        except ImportError:
            return "Web search is unavailable: the search package is not installed."
        try:
            results = list(DDGS().text(query, max_results=max_results))
        except Exception as error:
            return f"DuckDuckGo search failed: {error}"
        if not results:
            return "No results found."
        return "\n\n".join(
            f"[{item.get('title', '')}]({item.get('href', '')})\n{item.get('body', '')[:400]}" for item in results)

    return StructuredTool.from_function(
        coroutine=_run, name="web_search", args_schema=SearchInput,
        description="Search the web with DuckDuckGo. Use for current information, news, prices or anything you are unsure about. Needs no API key.")


def create_tavily_tool(config):
    api_key = (config.get("api_key") or "").strip()
    if not api_key:
        return None
    max_results = int(config.get("max_results") or 5)

    async def _run(query: str):
        data, error = await _request(
            "POST", "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": max_results, "search_depth": "basic"})
        if error:
            return f"Tavily search failed: {error}"
        results = (data or {}).get("results", []) if isinstance(data, dict) else []
        if not results:
            return "No results found."
        answer = (data or {}).get("answer")
        lines = [f"[{r.get('title', '')}]({r.get('url', '')})\n{r.get('content', '')[:500]}" for r in results]
        return (f"{answer}\n\n" if answer else "") + "\n\n".join(lines)

    return StructuredTool.from_function(
        coroutine=_run, name="tavily_search", args_schema=SearchInput,
        description="Search the web with Tavily, which returns cleaner summaries than a plain search engine.")


def create_serper_tool(config):
    api_key = (config.get("api_key") or "").strip()
    if not api_key:
        return None
    max_results = int(config.get("max_results") or 5)

    async def _run(query: str):
        data, error = await _request(
            "POST", "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": max_results})
        if error:
            return f"Google search failed: {error}"
        organic = (data or {}).get("organic", []) if isinstance(data, dict) else []
        if not organic:
            return "No results found."
        return "\n\n".join(
            f"[{r.get('title', '')}]({r.get('link', '')})\n{r.get('snippet', '')}" for r in organic[:max_results])

    return StructuredTool.from_function(
        coroutine=_run, name="google_search", args_schema=SearchInput,
        description="Search Google (via Serper) for current information on the web.")


class WebFetchInput(BaseModel):
    url: str = Field(description="Full URL of the page to read, including https://")


def create_web_fetch_tool(_config):
    async def _run(url: str):
        if not url.startswith(("http://", "https://")):
            return "The URL must start with http:// or https://"
        body, error = await _request("GET", url, headers={"User-Agent": "Mozilla/5.0 (SimpleAgent)"})
        if error:
            return f"Could not fetch the page: {error}"
        if isinstance(body, (dict, list)):
            return json.dumps(body)[:6000]
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", body or "", flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()[:6000] or "The page returned no readable text."

    return StructuredTool.from_function(
        coroutine=_run, name="read_web_page", args_schema=WebFetchInput,
        description="Fetch a URL and return its readable text. Use to read an article or page, including one a search returned.")


class HttpRequestInput(BaseModel):
    url: str = Field(description="Full URL to call, including https://")
    method: str = Field(default="GET", description="GET, POST, PUT, PATCH or DELETE")
    body: dict = Field(default_factory=dict, description="JSON body for POST/PUT/PATCH")


def create_http_tool(config):
    extra_headers = config.get("headers") if isinstance(config.get("headers"), dict) else {}

    async def _run(url: str, method: str = "GET", body: dict = None):
        if not url.startswith(("http://", "https://")):
            return "The URL must start with http:// or https://"
        method = (method or "GET").upper()
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            return f"Unsupported method: {method}"
        kwargs = {"headers": {"Content-Type": "application/json", **extra_headers}}
        if method in {"POST", "PUT", "PATCH"}:
            kwargs["json"] = body or {}
        data, error = await _request(method, url, **kwargs)
        if error:
            return error
        return json.dumps(data)[:6000] if isinstance(data, (dict, list)) else str(data)[:6000]

    return StructuredTool.from_function(
        coroutine=_run, name="http_request", args_schema=HttpRequestInput,
        description="Call any JSON HTTP API. Use when the user asks to hit an endpoint no other tool covers.")


# ── reference data (no API key) ─────────────────────────────────────────────

class WikipediaInput(BaseModel):
    topic: str = Field(description="The subject to look up, e.g. 'Ada Lovelace' or 'photosynthesis'")


def create_wikipedia_tool(_config):
    async def _run(topic: str):
        data, error = await _request(
            "GET", "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": topic, "srlimit": "1", "format": "json"},
            headers={"User-Agent": "SimpleAgent/1.0"})
        if error:
            return f"Wikipedia lookup failed: {error}"
        hits = (((data or {}).get("query") or {}).get("search") or []) if isinstance(data, dict) else []
        if not hits:
            return f'Wikipedia has no article matching "{topic}".'
        title = hits[0]["title"]
        summary, error = await _request(
            "GET", f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title, safe='')}",
            headers={"User-Agent": "SimpleAgent/1.0"})
        if error or not isinstance(summary, dict):
            return f"Found the article '{title}' but could not read it: {error}"
        return json.dumps({
            "title": summary.get("title"), "summary": summary.get("extract", "")[:3000],
            "url": ((summary.get("content_urls") or {}).get("desktop") or {}).get("page", ""),
        })

    return StructuredTool.from_function(
        coroutine=_run, name="wikipedia_lookup", args_schema=WikipediaInput,
        description="Look up factual background on a person, place, organisation or concept from Wikipedia. Needs no API key.")


class CurrencyInput(BaseModel):
    amount: float = Field(default=1, description="How much to convert")
    from_currency: str = Field(description="Three-letter source currency code, e.g. USD")
    to_currency: str = Field(description="Three-letter target currency code, e.g. INR")


def create_currency_tool(_config):
    async def _run(from_currency: str, to_currency: str, amount: float = 1):
        source, target = from_currency.strip().upper(), to_currency.strip().upper()
        if not (len(source) == 3 and len(target) == 3):
            return "Currency codes must be three letters, e.g. USD or INR."
        data, error = await _request(
            "GET", "https://api.frankfurter.app/latest",
            params={"amount": str(amount), "from": source, "to": target})
        if error:
            return f"Exchange rate lookup failed: {error}"
        rates = (data or {}).get("rates", {}) if isinstance(data, dict) else {}
        if target not in rates:
            return f"No exchange rate is published for {source} to {target}."
        return json.dumps({"amount": amount, "from": source, "to": target,
                           "converted": rates[target], "rate_date": (data or {}).get("date")})

    return StructuredTool.from_function(
        coroutine=_run, name="currency_convert", args_schema=CurrencyInput,
        description="Convert an amount between currencies at today's published rate. Needs no API key.")


# ── messaging and code hosting ──────────────────────────────────────────────

class SlackInput(BaseModel):
    message: str = Field(description="The message text to post")
    channel: str = Field(default="", description='Channel such as "#general". Only used with a bot token.')


def create_slack_tool(config):
    webhook_url = (config.get("webhook_url") or "").strip()
    bot_token = (config.get("api_key") or "").strip()
    default_channel = (config.get("channel") or "").strip()
    if not webhook_url and not bot_token:
        return None

    async def _run(message: str, channel: str = ""):
        target = (channel or default_channel).strip()
        if webhook_url:
            payload = {"text": message}
            if target:
                payload["channel"] = target
            _, error = await _request("POST", webhook_url, json=payload)
            return error or f'Message posted to Slack{f" ({target})" if target else ""}.'
        if not target:
            return "A channel is required when posting with a bot token."
        data, error = await _request(
            "POST", "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {bot_token}", "Content-Type": "application/json"},
            json={"channel": target, "text": message})
        if error:
            return error
        if isinstance(data, dict) and not data.get("ok"):
            return f"Slack rejected the message: {data.get('error', 'unknown error')}"
        return f"Message posted to {target}."

    return StructuredTool.from_function(
        coroutine=_run, name="slack_post_message", args_schema=SlackInput,
        description="Post a message to Slack. Use when asked to notify, send or share something on Slack.")


class GithubInput(BaseModel):
    action: str = Field(description="One of: list_issues, get_issue, create_issue, list_commits, get_repo")
    title: str = Field(default="", description="Issue title, for create_issue")
    body: str = Field(default="", description="Issue body, for create_issue")
    issue_number: int = Field(default=0, description="Issue number, for get_issue")


def create_github_tool(config):
    token = (config.get("api_key") or "").strip()
    repo = (config.get("repo") or "").strip()
    if not token or not repo:
        return None
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    base = f"https://api.github.com/repos/{repo}"

    async def _run(action: str, title: str = "", body: str = "", issue_number: int = 0):
        if action == "list_issues":
            data, error = await _request("GET", f"{base}/issues?per_page=20", headers=headers)
            if error:
                return error
            return json.dumps([{"number": i.get("number"), "title": i.get("title"), "state": i.get("state")}
                               for i in data or [] if isinstance(i, dict)])
        if action == "get_issue":
            data, error = await _request("GET", f"{base}/issues/{issue_number}", headers=headers)
            return error or json.dumps({k: data.get(k) for k in ("number", "title", "state", "body")})
        if action == "create_issue":
            if not title:
                return "A title is required to create an issue."
            data, error = await _request("POST", f"{base}/issues", headers=headers, json={"title": title, "body": body})
            return error or f"Created issue #{data.get('number')}: {data.get('html_url')}"
        if action == "list_commits":
            data, error = await _request("GET", f"{base}/commits?per_page=20", headers=headers)
            if error:
                return error
            return json.dumps([{"sha": (c.get("sha") or "")[:8], "message": (c.get("commit") or {}).get("message", "")}
                               for c in data or [] if isinstance(c, dict)])
        if action == "get_repo":
            data, error = await _request("GET", base, headers=headers)
            return error or json.dumps({k: data.get(k) for k in
                                        ("full_name", "description", "stargazers_count", "open_issues_count")})
        return f"Unknown action: {action}"

    return StructuredTool.from_function(
        coroutine=_run, name="github", args_schema=GithubInput,
        description=f"Read and write issues, commits and metadata on the GitHub repository {repo}.")


# ── catalogue ───────────────────────────────────────────────────────────────
#
# `config_fields` are what the user fills in when adding the tool and `requires`
# lists the ones without which the factory returns None, so the UI can say up
# front which tools need a credential.

BUILTIN_TOOLS = {
    "datetime": {
        "name": "Date & Time", "description": "Current date, time and day of the week",
        "category": "Utility", "config_fields": ["timezone_offset_hours"], "requires": [],
        "factory": create_datetime_tool,
    },
    "calculator": {
        "name": "Calculator", "description": "Exact arithmetic, so the model never does mental maths",
        "category": "Utility", "config_fields": [], "requires": [], "factory": create_calculator_tool,
    },
    "web_search": {
        "name": "Web Search", "description": "DuckDuckGo results — no API key needed",
        "category": "Web", "config_fields": ["max_results"], "requires": [], "factory": create_duckduckgo_tool,
    },
    "wikipedia": {
        "name": "Wikipedia", "description": "Factual background on a topic — no API key needed",
        "category": "Reference", "config_fields": [], "requires": [], "factory": create_wikipedia_tool,
    },
    "currency": {
        "name": "Currency Converter", "description": "Convert between currencies at today's rate",
        "category": "Reference", "config_fields": [], "requires": [], "factory": create_currency_tool,
    },
    "web_fetch": {
        "name": "Web Page Reader", "description": "Fetch a URL and read its text",
        "category": "Web", "config_fields": [], "requires": [], "factory": create_web_fetch_tool,
    },
    "http_request": {
        "name": "HTTP Request", "description": "Call any JSON API endpoint",
        "category": "Web", "config_fields": ["headers"], "requires": [], "factory": create_http_tool,
    },
    "tavily": {
        "name": "Tavily Search", "description": "Search results summarised for LLMs",
        "category": "Web", "config_fields": ["api_key", "max_results"], "requires": ["api_key"],
        "factory": create_tavily_tool,
    },
    "google_search": {
        "name": "Google Search (Serper)", "description": "Google results through the Serper API",
        "category": "Web", "config_fields": ["api_key", "max_results"], "requires": ["api_key"],
        "factory": create_serper_tool,
    },
    "slack": {
        "name": "Slack", "description": "Post a message to a Slack channel",
        "category": "Action", "config_fields": ["webhook_url", "api_key", "channel"],
        "requires": ["webhook_url|api_key"], "factory": create_slack_tool,
    },
    "github": {
        "name": "GitHub", "description": "Issues, commits and repository metadata",
        "category": "Action", "config_fields": ["api_key", "repo"], "requires": ["api_key", "repo"],
        "factory": create_github_tool,
    },
    "document_search": {
        "name": "Document Search", "description": "Search the documents you uploaded (vector retrieval)",
        "category": "Retrieval", "config_fields": ["top_k"], "requires": [], "factory": None,
    },
}


def create_builtin(tool_type, config):
    entry = BUILTIN_TOOLS.get(tool_type)
    if not entry or not entry["factory"]:
        return None
    try:
        return entry["factory"](config or {})
    except Exception:
        return None
