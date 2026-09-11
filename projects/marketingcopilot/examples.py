"""Ready-made scenarios.

The copilot arrives with a corpus but no idea what to ask it, and a blank chat
box is a poor way to discover that the same system answers four quite different
kinds of question by four quite different routes. Typing a question and getting
a good answer teaches you that it works; it does not teach you *how* it works.

So each scenario below is chosen to exercise one route and make one point, and
they are ordered so that running them in sequence builds the picture: a document
lookup, then a database query, then the hybrid question that needs both, then a
generated asset that gets compliance-checked, and finally a question with no
answer in the corpus — because a system that never refuses is not accurate, only
confident.

Each carries follow-ups as well as an opening question. They are not padding:
the follow-ups exercise multi-turn behaviour, where the question has to be
resolved against the conversation before it can be retrieved at all.
"""

EXAMPLES = [
    {
        "id": "brand-lookup",
        "order": 1,
        "title": "Look something up",
        "tagline": "One retrieval, one answer, with sources",
        "route": "rag",
        "difficulty": "start here",
        "what_it_shows": (
            "The simplest path through the system, and the one most questions take. No agent "
            "loop, no tools — the question is classified, the documents are searched, and the "
            "answer is written from what came back."
        ),
        "look_for": (
            "The citations under the answer. Every claim should trace to a document you can open "
            "in the Corpus tab, and the route chip should read “documents”. Note how fast it is "
            "compared with the hybrid scenario: this is what the routing buys you."
        ),
        "question": "What tone should we use in fintech campaigns?",
        "follow_ups": [
            "Which words are we not allowed to use?",
            "How is that different from how we talk to banking customers?",
        ],
    },
    {
        "id": "the-numbers",
        "order": 2,
        "title": "Ask for a number",
        "tagline": "Text to SQL, with the query shown",
        "route": "sql",
        "difficulty": "start here",
        "what_it_shows": (
            "Anything comparative or arithmetic is routed away from the documents entirely and "
            "answered with a database query. Vector search retrieves; it does not compute — and a "
            "system that tries to do arithmetic over retrieved text produces a fluent wrong number."
        ),
        "look_for": (
            "The SQL under the answer. It is the query that actually ran, shown so you can check "
            "it in five seconds. Try the third question: events look catastrophic on cost per "
            "acquisition, and the events playbook explains why that comparison is the wrong one."
        ),
        "question": "Which channel had the lowest cost per acquisition in Q3 2024?",
        "follow_ups": [
            "What was total spend on LinkedIn that quarter?",
            "Compare cost per acquisition across every channel for fintech.",
        ],
    },
    {
        "id": "diagnose",
        "order": 3,
        "title": "Diagnose a campaign",
        "tagline": "The question the whole architecture exists for",
        "route": "hybrid",
        "difficulty": "the interesting one",
        "what_it_shows": (
            "A question that cannot be answered by either half alone. The numbers say the campaign "
            "underperformed; only the documents say why. The agent fetches both and reconciles "
            "them, which is the honest justification for using an agent at all."
        ),
        "look_for": (
            "Expand “How it got there”. You should see the router choose hybrid, a retrieval step, "
            "a grader verdict and a SQL step — and the answer should separate what the data shows "
            "from what the retro claims. The retro also notes this was the third time the same "
            "mistake was made, which is exactly the institutional memory the project is for."
        ),
        "question": "Why did the Q3 LinkedIn fintech campaign underperform?",
        "follow_ups": [
            "How did Q2 compare, and what was different about it?",
            "Has this happened before?",
        ],
    },
    {
        "id": "draft-copy",
        "order": 4,
        "title": "Draft something, and have it checked",
        "tagline": "Brand voice retrieved, claims checked before you see it",
        "route": "generate",
        "difficulty": "the useful one",
        "what_it_shows": (
            "Generation grounded in the brand guide rather than in the model's general sense of "
            "marketing copy, followed by a compliance pass over the result. The check runs on "
            "everything the copilot drafts, before it reaches you."
        ),
        "look_for": (
            "The compliance panel under the draft. Then try the third question, which asks for "
            "something the rules forbid — the draft should either avoid the claim or be flagged "
            "with the rule it broke. The node reports rather than silently rewriting, because a "
            "guardrail that hides what it caught teaches nobody anything."
        ),
        "question": "Draft a LinkedIn ad for the fintech segment about automated controls testing.",
        "follow_ups": [
            "Make it shorter and lead with the customer's problem instead.",
            "Now write one claiming we're the fastest solution on the market.",
        ],
    },
    {
        "id": "the-refusal",
        "order": 5,
        "title": "Ask something it cannot know",
        "tagline": "The scenario where the right answer is “I don't know”",
        "route": "rag",
        "difficulty": "the honest one",
        "what_it_shows": (
            "Abstention. The corpus has nothing about TikTok or Japan, and the correct behaviour "
            "is to say so and name what it did search — not to assemble a plausible strategy from "
            "general knowledge, which is precisely what an ungrounded system would do here."
        ),
        "look_for": (
            "The grader sending retrieval back for a second attempt before giving up, and the "
            "attempt counter on the answer. The loop is capped at two on purpose: an unbounded "
            "self-correction loop is a bill and an outage. This behaviour is tested — two cases in "
            "the Evaluate tab exist only to check that it still refuses."
        ),
        "question": "What is our TikTok strategy for the Japanese market?",
        "follow_ups": [
            "What was our spend in Q1 2019?",
            "So what do you actually know about?",
        ],
    },
]

EXAMPLES_BY_ID = {example["id"]: example for example in EXAMPLES}


def example_by_id(example_id: str) -> dict | None:
    return EXAMPLES_BY_ID.get(example_id)
