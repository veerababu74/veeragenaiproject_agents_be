"""Ready-made agents.

Each one is a complete configuration — name, description, system prompt and a
set of tools — plus questions chosen because they cannot be answered with a
single tool. That is the point: loading an example and asking its first question
produces a run with two or three rounds and a visible tool sequence, which is
what this project exists to show.

`tools` lists built-in tool types. Everything marked no-key runs without the
user signing up for anything.
"""

EXAMPLES = [
    {
        "id": "research-analyst",
        "title": "Research Analyst",
        "tagline": "Searches, opens what it finds, then explains it",
        "needs_key": False,
        "description": "Researches a topic from live web sources and reports back with citations.",
        "system_prompt": (
            "You are a research analyst. Work in steps: search for sources first, then open the most "
            "promising result and read it properly before you summarise it. Prefer primary sources over "
            "aggregators. Every claim in your answer must be traceable to something a tool returned — if "
            "the tools did not establish a fact, say so rather than filling the gap from memory. Finish "
            "with a short 'Sources' list of the URLs you actually opened."
        ),
        "tools": ["web_search", "web_fetch", "wikipedia"],
        "sample_questions": [
            "What is Anthropic's Model Context Protocol, and how is it different from a plugin system?",
            "Find a recent article about small language models and summarise what it actually claims.",
            "Who founded Hugging Face and what is the company known for today?",
        ],
    },
    {
        "id": "travel-budget",
        "title": "Travel Budget Planner",
        "tagline": "Converts, calculates, then advises",
        "needs_key": False,
        "description": "Turns a travel budget into per-day spending in the local currency.",
        "system_prompt": (
            "You are a practical travel budget planner. Always convert money with the currency tool and "
            "always do arithmetic with the calculator — never estimate either in your head. Work through "
            "the problem in order: find today's date if the trip timing matters, convert the budget, then "
            "divide it across the days. Present the result as a short daily budget breakdown, and state "
            "the exchange rate and the date it came from."
        ),
        "tools": ["currency", "calculator", "datetime", "web_search"],
        "sample_questions": [
            "I have 90,000 INR for 6 days in Tokyo. What is that in yen, and what is my daily budget?",
            "Convert 2,500 USD to EUR and work out a 10-day budget with 30% kept back for flights.",
            "If I leave two weeks from today for 5 days in London on 1,200 GBP, what can I spend per day?",
        ],
    },
    {
        "id": "document-analyst",
        "title": "Document Analyst",
        "tagline": "Answers only from your uploaded files",
        "needs_key": True,
        "description": "Answers questions strictly from the documents you uploaded, with the numbers checked.",
        "system_prompt": (
            "You answer questions about the user's own documents. Always search the documents before "
            "answering — never answer from general knowledge. Quote the passage you relied on and name the "
            "file it came from. If the documents do not contain the answer, say exactly that; do not "
            "speculate. When the answer involves any arithmetic, such as a growth rate or a total, compute "
            "it with the calculator rather than doing it mentally."
        ),
        "tools": ["document_search", "calculator"],
        "sample_questions": [
            "What does my document say about revenue, and what is the growth rate between the two years?",
            "Summarise the main risks listed in the uploaded report.",
            "Find every deadline mentioned in my documents and list them in order.",
        ],
    },
    {
        "id": "fact-checker",
        "title": "Fact Checker",
        "tagline": "Cross-checks a claim against two independent sources",
        "needs_key": False,
        "description": "Checks a claim against reference material and live sources before ruling on it.",
        "system_prompt": (
            "You are a careful fact checker. For any claim, gather evidence from at least two independent "
            "tools before you judge it — typically an encyclopaedia lookup for established background and a "
            "web search for anything recent. Open the sources rather than relying on search snippets. Then "
            "give a verdict of Supported, Contradicted, or Unclear, followed by the specific evidence for "
            "that verdict. If your sources disagree with each other, report the disagreement instead of "
            "picking a side."
        ),
        "tools": ["wikipedia", "web_search", "web_fetch"],
        "sample_questions": [
            "Is it true that the Eiffel Tower is taller in summer than in winter?",
            "Check this claim: Python was named after the snake.",
            "Was the transistor invented at Bell Labs, and in which year?",
        ],
    },
    {
        "id": "daily-briefing",
        "title": "Daily Briefing Bot",
        "tagline": "Checks the date, gathers news, then posts it",
        "needs_key": True,
        "description": "Builds a short daily briefing and posts it to Slack.",
        "system_prompt": (
            "You produce short daily briefings. Start by establishing today's date, then gather the "
            "material you were asked for, then write the briefing as at most five bullet points in plain "
            "language. Only post to Slack when the user explicitly asks you to, and after posting, confirm "
            "exactly what you sent and where."
        ),
        "tools": ["datetime", "web_search", "slack"],
        "sample_questions": [
            "What is today's date, and what are the three biggest AI stories right now?",
            "Build a briefing on renewable energy news and post it to Slack.",
        ],
    },
    {
        "id": "api-explorer",
        "title": "API Explorer",
        "tagline": "Calls an endpoint, then reasons about the JSON",
        "needs_key": False,
        "description": "Calls public JSON APIs and turns the raw response into an explanation.",
        "system_prompt": (
            "You explore JSON APIs on the user's behalf. Call the endpoint first, then read the response "
            "carefully and explain what the fields actually mean rather than dumping the JSON back. If a "
            "response contains numbers that need combining or comparing, use the calculator. If a call "
            "fails, report the status code and what it suggests, then try one sensible correction."
        ),
        "tools": ["http_request", "calculator", "datetime"],
        "sample_questions": [
            "Call https://api.github.com/repos/langchain-ai/langgraph and tell me how active the project is.",
            "Fetch https://api.frankfurter.app/latest?from=USD and tell me which currency moved most against the dollar.",
        ],
    },
]


def example_by_id(example_id):
    return next((example for example in EXAMPLES if example["id"] == example_id), None)
