"""The written explanation, served alongside the working system.

This project is a demonstration as much as a tool, so the reasoning is shipped
next to the behaviour: what each node does, why the router exists, what every
metric means. The pattern follows the rest of the platform — a formula, the
symbols in it, and the operation decomposed — because a number nobody can
interpret is a number nobody acts on.
"""

OVERVIEW = {
    "title": "Marketing Copilot",
    "subtitle": "RAG and an agent over a marketing team's documents, numbers and rules",
    "problem": (
        "A marketing team's knowledge sits in three places that cannot talk to each other. "
        "Brand guidelines, campaign briefs and retros are documents nobody can find. Campaign "
        "performance is in a database only analysts can query. Claim rules are in a PDF people "
        "meet only when legal rejects their copy."),
    "why_it_needs_an_agent": (
        "One question decides the architecture: 'why did the Q3 LinkedIn fintech campaign "
        "underperform, and draft a revised brief'. That needs a SQL query for the numbers, a "
        "document search for the context, a synthesis of both, a generated asset, and a "
        "compliance check on what it wrote. The number and order of those steps depends on the "
        "question, which is the honest reason to reach for an agent rather than a chain."),
    "what_stays_dumb": (
        "Most questions are not that. 'What is our brand voice' is one retrieval and one "
        "generation; 'what did we spend on LinkedIn' is one tool call. The routing is the "
        "intelligent part so the common paths can stay cheap — an agent loop is latency, cost "
        "and a new way to fail, and it is worth paying for only where it earns its place."),
    "stack": [
        {"name": "FastAPI", "role": "HTTP surface, auth, background indexing",
         "why": "The work is IO-bound — waiting on the model, the vector store and SQLite — "
                "so async lets one worker hold many conversations at once."},
        {"name": "LangGraph", "role": "the agent, as an explicit state machine",
         "why": "Every node is a function you can test, the control flow is data rather than "
                "prose in a prompt, and the path a request took is recoverable afterwards."},
        {"name": "LangChain", "role": "model and embedding adapters",
         "why": "Used for plumbing, not orchestration. Swapping provider is a config change."},
        {"name": "Pinecone", "role": "vector store, one namespace per workspace",
         "why": "Metadata filtering pushes the quarter/channel/segment constraint into the "
                "query. Falls back to brute-force cosine over SQLite when unconfigured."},
        {"name": "SQLite", "role": "campaign data and every trace",
         "why": "Business data and observability in one database, so 'which answers got a "
                "thumbs-down and what did they cite' is a join rather than a project."},
    ],
}

NODES = [
    {
        "id": "route", "order": 1, "name": "Route",
        "tagline": "Decide which path the question takes",
        "formula": "route ∈ {rag, sql, hybrid, generate}",
        "symbols": [
            {"symbol": "rag", "means": "documented knowledge — brand, briefs, playbooks, retros"},
            {"symbol": "sql", "means": "numbers — spend, CTR, CAC, ROAS, comparisons"},
            {"symbol": "hybrid", "means": "needs both: any question asking why"},
            {"symbol": "generate", "means": "write an asset"},
        ],
        "derivation": [
            {"label": "Classify", "expression": "small model, temperature 0, one word out",
             "note": "A classifier that varies between identical inputs cannot be evaluated, "
                     "so this node is the coldest in the graph."},
            {"label": "Extract filters", "expression": "quarter, channel, segment ← question",
             "note": "Anything that would be a SQL WHERE clause becomes a retrieval filter "
                     "rather than a hope that the embedding encoded it."},
            {"label": "Fall back", "expression": "ambiguous → hybrid",
             "note": "Doing both is slower; being wrong is worse. A misroute makes every "
                     "downstream measurement meaningless."},
        ],
        "why": "Most traffic does not need an agent. Making the routing smart is what lets the "
               "common paths stay a single cheap call.",
    },
    {
        "id": "retrieve", "order": 2, "name": "Retrieve",
        "tagline": "Filter first, then hybrid search, then fuse",
        "formula": "RRF(dense, lexical) = Σ 1 / (k + rank)",
        "symbols": [
            {"symbol": "dense", "means": "cosine similarity over embeddings — meaning"},
            {"symbol": "lexical", "means": "term overlap — exact tokens like 2024-Q3"},
            {"symbol": "k", "means": "a smoothing constant, 60 by convention"},
            {"symbol": "rank", "means": "position in one retriever's ranking, from 1"},
        ],
        "derivation": [
            {"label": "Pre-filter", "expression": "WHERE quarter = '2024-Q3' AND channel = 'linkedin'",
             "note": "Two campaign briefs are near-identical semantically and differ by a date "
                     "the embedding barely encodes. Filtering makes it a lookup, not a hope."},
            {"label": "Dense", "expression": "cos(q, c) = (q · c) / (‖q‖ ‖c‖)",
             "note": "Meaning, independent of wording."},
            {"label": "Lexical", "expression": "Σ count(term) / √|chunk|",
             "note": "Catches the exact tokens dense retrieval blurs. Length-normalised so a "
                     "long chunk cannot win on volume."},
            {"label": "Fuse", "expression": "score = Σ 1 / (60 + rank)",
             "note": "Rank-based on purpose: the two scores are on different scales, and adding "
                     "them directly would just mean the larger scale wins every time."},
        ],
        "why": "Marketing text is full of exact identifiers — campaign codes, quarters, channel "
               "names — and dense retrieval alone is measurably poor at those.",
    },
    {
        "id": "grade", "order": 3, "name": "Grade the context",
        "tagline": "Does what came back actually answer the question?",
        "formula": "relevant? → synthesise : rewrite (while attempts < 2)",
        "symbols": [
            {"symbol": "attempts", "means": "retrieval rewrites used so far"},
            {"symbol": "2", "means": "the bound — chosen, not tuned"},
        ],
        "derivation": [
            {"label": "Judge the context, not the answer",
             "expression": "can these passages answer the question? YES / NO",
             "note": "Catching a retrieval failure before an answer is written is the point. "
                     "Grading afterwards means the fluent wrong answer already exists."},
            {"label": "Rewrite and retry", "expression": "reformulate → retrieve again",
             "note": "Different wording, broader terms. This is the CRAG pattern."},
            {"label": "Stop", "expression": "attempts = 2 → abstain",
             "note": "An unbounded self-correction loop is a bill and an outage. The behaviour "
                     "at the bound is defined: say what was not found."},
        ],
        "why": "Bad retrieval does not raise an exception. It returns confident nonsense with a "
               "200, and this node is the only thing in the graph looking for that.",
    },
    {
        "id": "sql", "order": 4, "name": "Query the numbers",
        "tagline": "Text to SQL, validated and read-only",
        "formula": "CAC = spend / conversions   ROAS = pipeline / spend   CTR = clicks / impressions",
        "symbols": [
            {"symbol": "CAC", "means": "cost per acquisition"},
            {"symbol": "ROAS", "means": "return on ad spend"},
            {"symbol": "CTR", "means": "click-through rate"},
        ],
        "derivation": [
            {"label": "Generate", "expression": "schema + worked examples → one SELECT"},
            {"label": "Validate", "expression": "single SELECT, no writes, campaigns only, LIMIT imposed",
             "note": "Parsed and rejected before execution, not after."},
            {"label": "Execute read-only", "expression": "sqlite3 file:…?mode=ro",
             "note": "The database refuses a write. That is the control that matters — "
                     "validation can be outsmarted by a clever string, a read-only file "
                     "descriptor cannot."},
            {"label": "Show the query", "expression": "returned with the answer",
             "note": "An analyst can check it in five seconds. A number nobody can check is a "
                     "number nobody should paste into a deck."},
        ],
        "why": "Vector search retrieves; it does not compute. Asked for ROAS, a retrieval-only "
               "system finds chunks mentioning ROAS and does confident mental arithmetic.",
    },
    {
        "id": "synthesise", "order": 5, "name": "Answer",
        "tagline": "Write from what was gathered, or refuse",
        "formula": "answer = f(context, rows) | abstention",
        "symbols": [
            {"symbol": "context", "means": "the retrieved chunks that survived grading"},
            {"symbol": "rows", "means": "whatever SQL returned"},
        ],
        "derivation": [
            {"label": "Ground", "expression": "use only the material supplied",
             "note": "Never invent a number, a customer name or a date."},
            {"label": "Cite", "expression": "[1], [2] matching the numbered passages",
             "note": "Marketers verify citations, so citation accuracy is measured too."},
            {"label": "Abstain", "expression": "nothing relevant → say so, and say what is covered",
             "note": "Has to be asked for explicitly. Left to itself a model will always find "
                     "something to say."},
        ],
        "why": "Abstention is a feature you build, not a behaviour you get. Two golden cases "
               "have no answer in the corpus and refusing them is the pass.",
    },
    {
        "id": "compliance", "order": 6, "name": "Compliance check",
        "tagline": "Every generated claim, against every rule",
        "formula": "violations = { rule : pattern(rule) matches draft }",
        "symbols": [
            {"symbol": "rule", "means": "one claim rule with a code and a severity"},
            {"symbol": "pattern", "means": "the phrasing that signals a violation"},
        ],
        "derivation": [
            {"label": "Load every rule", "expression": "all of them, never top-k",
             "note": "There are five. Retrieving the 'three most relevant' means the two it "
                     "skipped are the two that catch the violation."},
            {"label": "Match", "expression": "regex per rule, case-insensitive",
             "note": "Deterministic on purpose. A model judging its own output for compliance "
                     "is slower and less predictable than a pattern."},
            {"label": "Report, don't silently fix",
             "expression": "return the violation with the rule cited",
             "note": "It escalates rather than quietly rewriting, because a guardrail that "
                     "hides what it caught teaches nobody anything."},
        ],
        "why": "This is the node that makes the project marketing rather than generic. It is a "
               "guardrail, not a gate: it cuts what reaches legal, it does not replace legal.",
    },
]

FAILURE_MODES = [
    {"title": "The right topic, the wrong quarter",
     "symptom": "Asking about Q3 returned the Q1 brief.",
     "cause": "The date was prose inside the chunk. Two briefs are semantically near-identical "
              "and cosine similarity does not understand recency.",
     "fix": "Extract quarter, channel and segment at ingestion into metadata; parse them out of "
            "the question; apply as a pre-filter.",
     "lesson": "Anything you would put in a SQL WHERE clause belongs in metadata, not in the "
               "embedding."},
    {"title": "Confident arithmetic that was wrong",
     "symptom": "Asked for ROAS across channels, it produced a fluent, plausible, wrong number.",
     "cause": "Numeric aggregation routed to a vector store — a category error.",
     "fix": "Route anything comparative, aggregate or arithmetic to SQL, and show the query.",
     "lesson": "Vector search retrieves; it does not compute."},
    {"title": "The self-correction loop did not terminate",
     "symptom": "On an unanswerable question the grader rejected, the rewriter reformulated, "
                "retrieval failed again — forever.",
     "cause": "No bound, and no defined behaviour for 'the answer is not in the corpus'.",
     "fix": "Cap attempts at 2; on exhaustion return an explicit abstention naming what was "
            "searched; add unanswerable cases to the golden set so refusing is tested.",
     "lesson": "Every agent loop needs a bound and a defined behaviour at the bound."},
    {"title": "Duplicate chunks quietly degraded quality",
     "symptom": "Answers grew repetitive; the context filled with three copies of a paragraph.",
     "cause": "Re-indexing an edited document appended new chunks without removing the old.",
     "fix": "Content hash in the document registry; delete by document_id before upsert.",
     "lesson": "Retrieval bugs are usually ingestion bugs. Ingestion is a data pipeline and "
               "needs pipeline discipline."},
]
