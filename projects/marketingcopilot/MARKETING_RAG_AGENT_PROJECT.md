# Marketing Copilot — a RAG + agentic assistant

**An end-to-end project you can explain in an interview.**
Stack: FastAPI · LangChain · LangGraph · Pinecone · PostgreSQL · Python

---

## How to use this document

Read it in this order:

1. **The 90-second pitch** — memorise the shape, not the words.
2. **The problem** — this is what makes an interviewer lean in. Systems are boring; problems are interesting.
3. **Architecture → Evaluation → Monitoring** — the three sections that separate "I followed a tutorial" from "I have run this."
4. **What went wrong** — the single highest-value section. Interviewers hire people who have debugged things.
5. **The question bank** at the end.

> **One warning before you start.**
> Every number in this document is a **placeholder** — written as `‹your number›` or marked *illustrative*. Replace them with what you actually measured, or say "we tracked this, I don't remember the exact figure." An invented metric is the fastest way to fail an interview, because the follow-up question is always *"how did you measure that?"* and a fabricated number has no story behind it. A real number has a messy, convincing story behind it. Use the real one.

---

## Table of contents

- [The 90-second pitch](#the-90-second-pitch)
- [The problem](#the-problem)
- [Why RAG, and why agents](#why-rag-and-why-agents)
- [Architecture](#architecture)
- [The data model](#the-data-model)
- [The retrieval pipeline](#the-retrieval-pipeline)
- [The agent graph](#the-agent-graph)
- [The FastAPI service](#the-fastapi-service)
- [Evaluation](#evaluation)
- [Monitoring and observability](#monitoring-and-observability)
- [What went wrong, and what fixed it](#what-went-wrong-and-what-fixed-it)
- [Cost and latency](#cost-and-latency)
- [Security, tenancy and compliance](#security-tenancy-and-compliance)
- [How to explain this in an interview](#how-to-explain-this-in-an-interview)
- [Situational questions](#situational-questions)
- [Scenario questions](#scenario-questions)
- [Technical questions](#technical-questions)
- [Sixty-second answers to have ready](#sixty-second-answers-to-have-ready)

---

## The 90-second pitch

> "I built a marketing copilot for a B2B SaaS marketing team — about ‹40› people across demand gen, content, and product marketing.
>
> The problem was that their knowledge lived in three incompatible places. Brand guidelines, campaign briefs, and competitor battlecards were in documents nobody could find. Campaign performance was in Postgres, which only two analysts could query. And legal claim rules were in a PDF that people simply didn't read — so non-compliant copy kept reaching review and bouncing back.
>
> So a question like *'why did the Q3 LinkedIn campaign underperform, and draft me a revised brief'* needed a document search, a SQL query, and a compliance check — three different systems and usually three different people. It took days.
>
> I built a service that answers those questions in one place. It's a **LangGraph** agent behind **FastAPI**. A router classifies the question, then it either retrieves from **Pinecone**, generates read-only **SQL** against the performance database, or does both and synthesises. Every generated claim goes through a compliance node before it's returned, and every answer carries citations.
>
> The two parts I'd want to talk about most are **evaluation** — I built a golden set of ‹120› questions and measured retrieval and generation separately, because when quality drops you need to know *which half* broke — and **the self-correction loop**, where a grader node checks whether retrieved context actually answers the question and rewrites the query if it doesn't."

**Why this pitch works:** problem first, architecture second, and it ends by handing the interviewer two threads to pull on — both of which you're prepared for. Never end a pitch with "and that's it." End it by offering the next question.

---

## The problem

### The setting

A B2B SaaS company selling to financial services. Marketing has four functions: demand generation, content, product marketing, and campaign operations.

### The three silos

| Where knowledge lives | What's in it | Who can use it |
|---|---|---|
| **Documents** (Notion, Drive, PDFs) | Brand voice guide, 200+ campaign briefs, product one-pagers, competitor battlecards, ICP and persona docs, post-campaign retros | Anyone, in theory. Nobody, in practice — search is keyword-based and briefs are named `Brief_v3_FINAL_v2.docx` |
| **PostgreSQL** | Campaign spend, impressions, clicks, conversions, pipeline, CAC, ROAS by channel/segment/quarter | Two analysts. Everyone else files a ticket and waits ‹2–3› days |
| **The compliance PDF** | Which claims are permitted; what needs a disclaimer; what can never be said in a regulated vertical | Legal. Marketers discover the rules when their copy gets rejected |

### What this costs

Four concrete, measurable pains — these are what you're solving, and each one maps to a metric later:

1. **Analyst queue.** Simple questions like *"what was ROAS on paid search last quarter, by segment?"* consume analyst time that should go to real analysis.
2. **Rework loop.** Copy goes to legal, bounces back for a claim violation, gets rewritten, goes back. ‹2–3› round trips per asset.
3. **Onboarding.** A new marketer takes ‹6–8› weeks to learn brand voice and what's been tried before. Most of that is asking colleagues questions that a document already answers.
4. **Repeated mistakes.** Post-campaign retros exist but are never read, so the same failed audience targeting is tried again a year later.

### The question that defines the system

> *"Why did the Q3 LinkedIn campaign to the fintech segment underperform, and draft a revised brief for Q4."*

Look at what one sentence requires:

- **Numbers** — Q3 LinkedIn performance vs. benchmark, by segment → *SQL*
- **Context** — the original brief, the audience definition, the retro → *document retrieval*
- **Synthesis** — a causal explanation combining both
- **Generation** — a new brief in brand voice
- **Verification** — no non-compliant claims in the draft

**No single retrieval call answers this.** That is the honest justification for an agent, and it is the sentence to say out loud in an interview. Do not claim you needed an agent because agents are good. Claim it because the query decomposes into steps whose *number and order depend on the question*.

---

## Why RAG, and why agents

### Why RAG rather than fine-tuning

Say this crisply — it's a very common question:

- **The knowledge changes weekly.** New campaigns, new positioning, new competitor moves. Fine-tuning on a corpus that changes weekly means retraining weekly.
- **Attribution is a requirement, not a nice-to-have.** Marketers won't trust a claim about last quarter's ROAS without a source. RAG gives citations; a fine-tuned model gives confident prose.
- **Access control.** Some briefs are restricted. Retrieval can filter by permission at query time; weights cannot forget a document for one user and remember it for another.
- **Correcting an error is a document edit**, not a training run.

Fine-tuning would help with *form* — house tone, brief structure — not *facts*. If pushed: "I'd consider a small fine-tune or few-shot library for output formatting, and keep RAG for everything factual. They solve different problems."

### Why an agent rather than a fixed chain

Be honest about this, because over-claiming agents is a known interview smell:

| Query type | What it needs | Should it be agentic? |
|---|---|---|
| "What's our brand voice for LinkedIn?" | One retrieval, one generation | **No.** A fixed chain. Faster and cheaper. |
| "ROAS by channel last quarter?" | One SQL query | **No.** A tool call. |
| "Why did X underperform, draft a fix" | SQL + retrieval + synthesis + generation + compliance, order and count unknown in advance | **Yes.** |

So: **the router is the agent.** Most traffic (‹~70%› in this system) takes a cheap deterministic path. The agent loop exists for the minority of queries that genuinely need it.

> **Interview line:** "I'd push back on making everything agentic. An agent is a loop with a decision in it, and every loop is latency, cost, and a new failure mode. I made the *routing* smart so most queries could stay dumb."

That single sentence signals more seniority than any architecture diagram.

---

## Architecture

```
                        ┌──────────────────────────────┐
   Slack / Web UI ─────►│      FastAPI (async)         │
                        │  /chat (SSE stream)          │
                        │  /ingest  /feedback  /eval   │
                        │  JWT auth · tenant resolve   │
                        └──────────────┬───────────────┘
                                       │
                        ┌──────────────▼───────────────┐
                        │      LangGraph agent          │
                        │  state: messages, plan,       │
                        │  context, sql_result,         │
                        │  citations, attempts          │
                        └──┬──────┬──────┬──────┬───────┘
                           │      │      │      │
              ┌────────────▼┐ ┌───▼────┐ ┌▼─────────┐ ┌▼──────────┐
              │  retrieve   │ │  SQL   │ │ compliance│ │ synthesise│
              │  (Pinecone) │ │ (read- │ │  check    │ │  + cite   │
              │  + rerank   │ │  only) │ │           │ │           │
              └──────┬──────┘ └───┬────┘ └─────┬─────┘ └─────┬─────┘
                     │            │            │             │
              ┌──────▼────────────▼────────────▼─────────────▼──────┐
              │   Pinecone (namespaced)   │   PostgreSQL             │
              │   • doc chunks + metadata │   • campaign performance │
              │                           │   • conversations, msgs  │
              │                           │   • feedback, eval runs  │
              │                           │   • document registry    │
              └───────────────────────────┴──────────────────────────┘
                                       │
                        ┌──────────────▼───────────────┐
                        │ LangSmith traces · Prometheus │
                        │ structured logs · cost meter  │
                        └──────────────────────────────┘
```

### Component responsibilities

| Component | Job | Why this choice |
|---|---|---|
| **FastAPI** | HTTP surface, auth, streaming, background ingestion | Async suits an IO-bound workload — the service spends its time waiting on the LLM, Pinecone and Postgres. Native SSE for token streaming. Pydantic gives request validation for free. |
| **LangGraph** | The agent as an explicit state machine | Graphs are inspectable and testable. Each node is a pure-ish function I can unit test. Conditional edges make routing explicit rather than buried in a prompt. Checkpointing gives conversation memory and resumability. |
| **LangChain** | Loaders, splitters, retriever interfaces, LLM abstraction | I use it for the plumbing, not the orchestration. Swapping embedding providers is a config change. |
| **Pinecone** | Vector store, one namespace per tenant | Managed — no index infrastructure to run. Metadata filtering pushes access control into the query. Namespaces give hard tenant isolation. |
| **PostgreSQL** | Business data *and* application state | The campaign data was already there. Conversations, feedback and eval results live beside it so I can join "which answers got thumbs-down" against "which documents they cited." |

> **Note on Pinecone indexes:** one index per embedding model. A single index has a fixed dimension, so you cannot mix a 1536-dim and a 3072-dim model in one index. If you migrate embedding models, you build a new index and cut over — mention this if asked about model upgrades, it shows operational awareness.

---

## The data model

Keeping application state in SQL — not just business data — is what makes evaluation and monitoring possible later.

```sql
-- ── business data (already existed) ──────────────────────────────
CREATE TABLE campaigns (
    id              BIGSERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    channel         TEXT NOT NULL,         -- linkedin | paid_search | email | events
    segment         TEXT NOT NULL,         -- fintech | insurance | banking
    quarter         TEXT NOT NULL,
    start_date      DATE, end_date DATE,
    spend           NUMERIC(12,2),
    impressions     BIGINT,
    clicks          BIGINT,
    conversions     INTEGER,
    pipeline_value  NUMERIC(14,2)
);
CREATE INDEX ON campaigns (quarter, channel, segment);

-- ── the RAG side ─────────────────────────────────────────────────
CREATE TABLE documents (
    id            UUID PRIMARY KEY,
    tenant_id     UUID NOT NULL,
    title         TEXT NOT NULL,
    source_uri    TEXT NOT NULL,
    doc_type      TEXT NOT NULL,           -- brief | brand | battlecard | retro | compliance
    content_hash  TEXT NOT NULL,           -- skip re-embedding unchanged files
    chunk_count   INTEGER NOT NULL,
    indexed_at    TIMESTAMPTZ NOT NULL,
    UNIQUE (tenant_id, source_uri)
);

-- ── conversation + observability ─────────────────────────────────
CREATE TABLE conversations (
    id          UUID PRIMARY KEY,
    tenant_id   UUID NOT NULL,
    user_id     UUID NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE messages (
    id               UUID PRIMARY KEY,
    conversation_id  UUID REFERENCES conversations(id),
    role             TEXT NOT NULL,        -- user | assistant
    content          TEXT NOT NULL,
    route            TEXT,                 -- rag | sql | hybrid | generate
    citations        JSONB,                -- [{document_id, chunk_id, score}]
    tool_calls       JSONB,
    prompt_tokens    INTEGER,
    completion_tokens INTEGER,
    cost_usd         NUMERIC(10,6),
    latency_ms       INTEGER,
    trace_id         TEXT,                 -- ties this row to the LangSmith trace
    created_at       TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE feedback (
    id          UUID PRIMARY KEY,
    message_id  UUID REFERENCES messages(id),
    rating      SMALLINT NOT NULL,         -- +1 / -1
    reason      TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- ── evaluation ───────────────────────────────────────────────────
CREATE TABLE eval_runs (
    id           UUID PRIMARY KEY,
    git_sha      TEXT NOT NULL,
    dataset      TEXT NOT NULL,
    started_at   TIMESTAMPTZ, finished_at TIMESTAMPTZ,
    metrics      JSONB                     -- {recall_at_5, faithfulness, ...}
);
```

**The design point worth stating:** `messages.trace_id`, `messages.citations` and `feedback` together let me answer *"show me every thumbs-down answer from last week and the chunks it retrieved."* That query is how you find retrieval problems in production. Without it you are guessing.

---

## The retrieval pipeline

### Ingestion

```
source → load → clean → chunk → embed → upsert (Pinecone) → register (Postgres)
```

Run as a FastAPI **background task**, not inline — embedding 200 briefs takes minutes, and an HTTP request should not hold that.

**Chunking.** This is where most RAG quality is won or lost, so have a real opinion:

| Document type | Strategy | Why |
|---|---|---|
| Campaign briefs | **Structure-aware** — split on headings (Objective, Audience, Channels, Budget, Results), keep the section title in the chunk | A brief's headings *are* its semantics. "Audience" under one campaign must not merge with "Budget" of the next. |
| Brand guide | Recursive character, ~‹800› tokens, ~‹100› overlap | Continuous prose with no reliable structure. |
| Compliance rules | **One rule per chunk**, never split | A half-rule is worse than no rule. A chunk saying "you may claim market leadership" without "…if you cite the Gartner source" is actively dangerous. |
| Performance tables | Not embedded at all — routed to SQL | Embedding numbers is a category error. Vector similarity cannot do arithmetic, comparison or aggregation. |

> That last row is a strong interview point: **knowing what *not* to put in the vector store.** Many candidates embed everything.

**Metadata on every chunk** — this is what makes filtering possible:

```python
{
    "tenant_id":   "...",         # namespace-level isolation as well
    "document_id": "...",
    "doc_type":    "brief",       # filter: only search retros
    "channel":     "linkedin",
    "segment":     "fintech",
    "quarter":     "2024-Q3",
    "date":        1727740800,    # numeric, so range filters work
    "access":      "general",     # general | restricted
    "section":     "Audience",
}
```

### Query time

1. **Query understanding** — resolve pronouns against history ("draft a revised one" → "revised Q4 LinkedIn fintech brief"). Cheap model, big payoff on multi-turn.
2. **Metadata pre-filter** — "Q3 LinkedIn" becomes a filter, not a hope. Filtering before search beats hoping the embedding captures a date.
3. **Hybrid search** — dense (semantic) + sparse/BM25 (lexical). Marketing text is full of exact tokens: campaign codes, product names, `Q3-FS-LI-002`. Dense retrieval alone is poor at exact identifiers.
4. **Rerank** — over-retrieve `top_k=‹20›`, then a cross-encoder reranker cuts to `‹5›`. Biggest single quality win in the project, for one round trip.
5. **Context assembly** — pack under a token budget, newest-first for time-sensitive types, always carry the citation.

> **Interview line:** "Bi-encoders embed the query and document independently, so they never see them together. A cross-encoder scores the pair jointly — far more accurate, far too slow to run over the whole corpus. So you use the fast one to get 20 candidates and the accurate one to order them. That's the whole idea behind rerank."

---

## The agent graph

### State

```python
class AgentState(TypedDict):
    messages:     Annotated[list[BaseMessage], add_messages]
    question:     str
    route:        Literal["rag", "sql", "hybrid", "generate"]
    context:      list[Document]
    sql_query:    str | None
    sql_result:   list[dict] | None
    draft:        str | None
    citations:    list[dict]
    attempts:     int          # bounded — this is the loop guard
    compliance:   dict | None
```

### The graph

```
                    ┌─────────┐
     question ─────►│  route  │
                    └────┬────┘
         ┌───────────────┼───────────────┬──────────────┐
         ▼               ▼               ▼              ▼
    ┌─────────┐    ┌──────────┐   ┌───────────┐   ┌──────────┐
    │ retrieve│    │ sql_tool │   │  hybrid   │   │ generate │
    └────┬────┘    └────┬─────┘   │ (both)    │   └────┬─────┘
         │              │         └─────┬─────┘        │
         ▼              │               │              │
    ┌─────────┐         │               │              │
    │  grade  │         │               │              │
    │ context │         │               │              │
    └────┬────┘         │               │              │
    relevant? ──no──► ┌──────────┐      │              │
         │            │ rewrite  │──┐   │              │
         │yes         │  query   │  │   │              │
         │            └──────────┘  │   │              │
         │                 ▲────────┘   │              │
         │            (max 2 attempts)  │              │
         ▼                              ▼              ▼
    ┌────────────────────────────────────────────────────┐
    │              synthesise + attach citations          │
    └──────────────────────┬─────────────────────────────┘
                           ▼
                    ┌─────────────┐
                    │ compliance  │──violation──► revise (max 1)
                    └──────┬──────┘
                           ▼
                       respond
```

### The nodes that matter in an interview

**`route`** — a small, cheap model classifies into `rag | sql | hybrid | generate`. Structured output, not free text. Falls back to `hybrid` on low confidence, because being slow is better than being wrong.

**`grade_context`** — the self-correcting bit (this is the CRAG / Self-RAG pattern; name it, interviewers recognise it). An LLM judges each retrieved chunk: *does this actually help answer the question?* If nothing survives, `rewrite_query` reformulates and retries.

> **The critical detail:** `attempts` is bounded at 2. Say this unprompted. An unbounded self-correction loop is a bill and an outage. Bounding it, and having a defined behaviour at the bound ("I don't have enough information about X — here's what I did find"), is the difference between a demo and a service.

**`sql_tool`** — text-to-SQL, with guardrails that are the interesting part:
- Connects as a **read-only role** with grants on exactly the reporting tables. Not "the prompt says don't write" — the *database* says it can't.
- Schema and column descriptions in the prompt; few-shot examples of good queries.
- `LIMIT` injected; statement timeout set.
- Query is **validated before execution** — parse it, reject anything that isn't a single `SELECT`.
- Result returned as structured rows, and the **SQL is shown to the user** so an analyst can check it.

**`compliance`** — the domain-specific node, and the one that makes this project *marketing* rather than generic. Retrieves the applicable rules for the vertical, then checks the draft against them. Returns violations with the rule cited. One revision attempt, then it escalates to a human with the violation flagged rather than silently shipping.

> **Interview line:** "The compliance node is a guardrail, not a gate. It can't guarantee legal correctness and I never claimed it could. What it does is catch the obvious violations before a human reviewer sees them, which cut the review round trips."

---

## The FastAPI service

```python
@router.post("/chat")
async def chat(request: ChatRequest, user: User = Depends(current_user)):
    """Streams tokens over SSE. The graph runs as it streams."""

    async def events():
        config = {
            "configurable": {
                "thread_id": str(request.conversation_id),
                "tenant_id": str(user.tenant_id),
            },
            "callbacks": [tracer],
        }
        async for event in graph.astream_events(
            {"question": request.message, "attempts": 0},
            config=config, version="v2",
        ):
            match event["event"]:
                case "on_chat_model_stream":
                    yield sse("token", event["data"]["chunk"].content)
                case "on_chain_start":
                    # Node-level progress: "searching briefs...", "querying
                    # performance data...". A 12-second answer feels broken
                    # without it and fine with it.
                    yield sse("step", event["name"])
        yield sse("done", {"citations": ..., "trace_id": ...})

    return EventSourceResponse(events())
```

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat` | Streamed answer (SSE) |
| `POST` | `/ingest` | Register + index a document; returns a job id |
| `GET` | `/ingest/{job_id}` | Ingestion status |
| `POST` | `/feedback` | Thumbs up/down on a message |
| `GET` | `/conversations/{id}` | History |
| `POST` | `/eval/run` | Trigger an eval run against a dataset |
| `GET` | `/healthz` `/readyz` | Liveness / readiness |

### Design decisions worth defending

- **Async throughout.** The workload is IO-bound — waiting on LLM, Pinecone, Postgres. `async def` plus a connection pool means one worker serves many concurrent conversations. A blocking call in an async route poisons the event loop; anything CPU-bound goes to a thread pool.
- **Streaming, and streaming *steps*.** Perceived latency is the product. Time-to-first-token matters more than total time.
- **Idempotent ingestion** via `content_hash` — re-uploading an unchanged file is a no-op instead of duplicate chunks. Duplicate chunks are a silent, expensive retrieval-quality bug.
- **Tenant resolved from the JWT, never from the request body.** The client cannot ask for another tenant's namespace.
- **Timeouts and a circuit breaker** on every external call, with a degraded path: if Pinecone is down, answer from SQL and say retrieval is unavailable. Partial answers beat 500s.

---

## Evaluation

This is the section most candidates are weakest on, so it is your biggest opportunity.

### The principle

> **Evaluate retrieval and generation separately.** When answer quality drops, you must know whether you fetched the wrong context or wrote a bad answer from good context. One end-to-end score cannot tell you, and you will spend a day guessing.

### The golden dataset

‹120› questions, built from real Slack/ticket history — not invented. Each carries: the question, the document chunks that *should* be retrieved, a reference answer, and the expected route.

| Slice | Count | Why it's separate |
|---|---|---|
| Factual lookup | ‹40› | Pure retrieval quality |
| Analytical (SQL) | ‹30› | Text-to-SQL correctness |
| Hybrid multi-hop | ‹20› | The agent's reason to exist |
| Generative (drafting) | ‹20› | Voice and compliance |
| **Adversarial / unanswerable** | ‹10› | **Does it say "I don't know"?** |

That last slice is the one to mention. A system that never abstains is not a good system; it is a confident one.

### Retrieval metrics

| Metric | What it answers | Note |
|---|---|---|
| **Recall@k** | Did the right chunk make it into the top *k* at all? | The ceiling on everything downstream. If it isn't retrieved, no prompt can fix it. **Watch this one first.** |
| **Precision@k** | How much of what we retrieved was useful? | Low precision = wasted tokens + distraction |
| **MRR** | How high up was the first correct chunk? | Sensitive to ordering — the number that moves when you add rerank |
| **nDCG@k** | Ranking quality with graded relevance | When "relevant" isn't binary |
| **Context precision / recall** (RAGAS) | LLM-judged versions of the above | No hand labels needed; good for scale, noisier |

### Generation metrics

| Metric | What it answers |
|---|---|
| **Faithfulness / groundedness** | Is every claim supported by retrieved context? *The anti-hallucination metric.* |
| **Answer relevance** | Does it address the question actually asked? |
| **Correctness** | Versus the reference answer (LLM-as-judge, or exact match for numerics) |
| **Citation accuracy** | Do the citations point at the chunks the claims came from? Marketers verify these. |

### Agent metrics

These are what distinguish agent evaluation from RAG evaluation — most candidates have never thought about them:

| Metric | What it answers | Why it matters |
|---|---|---|
| **Routing accuracy** | Did it pick the right path? | A misroute makes every downstream metric meaningless |
| **Tool-selection accuracy** | Right tool, right arguments? | |
| **Task success rate** | Did the whole multi-step task complete correctly? | The number a business cares about |
| **Step efficiency** | Steps taken vs. minimum needed | Detects dithering loops |
| **Loop / termination rate** | How often does it hit the attempt cap? | A rising rate is an early warning of retrieval decay |
| **Recovery rate** | After a tool error, does it recover or collapse? | Inject failures deliberately to measure it |
| **Cost & latency per task** | p50/p95 tokens, dollars, seconds | Agents are unbounded by default |

### SQL-specific

**Execution accuracy** — run the generated SQL and the reference SQL, compare result sets. Far better than string-matching the query, because many different queries are correct.

### Business metrics — the ones that got it funded

| Metric | Baseline | Target |
|---|---|---|
| Analyst tickets for routine data pulls | ‹n/week› | ↓ ‹%› |
| Legal review round trips per asset | ‹2–3› | ↓ to ‹1› |
| Time-to-first-draft | ‹hours› | ↓ ‹%› |
| Human edit distance on drafts | — | tracked over time |
| New-hire ramp | ‹6–8 wks› | ↓ ‹%› |

> **Interview line:** "Faithfulness went up is an engineering result. Legal round trips went from three to one is a business result. I tracked both, and I led with the second when I talked to stakeholders."

### How it runs

- **Offline, in CI.** Every PR touching prompts, chunking or the graph runs the golden set. A regression beyond a threshold fails the build. Results land in `eval_runs` keyed by git SHA, so quality is diffable across commits.
- **Online.** Thumbs up/down joined to citations and traces. Weekly review of every negative — that queue is the roadmap.
- **LLM-as-judge, honestly.** It's noisy and biased toward verbosity. Mitigations: a fixed rubric, a strong judge model, position-swapping for pairwise comparisons, and a human-labelled subset to check the judge agrees with humans. State the limitation before the interviewer does.

---

## Monitoring and observability

Three layers. Say "three layers" — structure reads as experience.

### 1. Traces — LangSmith

Every request is one trace: route decision, retrieval (query, filters, chunks, scores), grader verdicts, SQL generated and rows returned, every LLM call with tokens and latency, compliance result. `trace_id` is stored on the message row, so a complaint in Slack becomes a trace in one query.

### 2. Metrics — Prometheus / Grafana

```
rag_requests_total{route, status}
rag_latency_seconds{stage}          # retrieve | rerank | sql | generate — histogram
rag_retrieved_score{quantile}       # top-1 similarity distribution
rag_grader_rejections_total         # context judged irrelevant
rag_loop_exhausted_total            # hit the attempt cap
rag_tokens_total{model, kind}
rag_cost_usd_total{model}
rag_compliance_violations_total
rag_feedback_total{rating}
```

**The three alerts that matter:**

| Alert | Why |
|---|---|
| `rag_retrieved_score` p50 **drops** | Retrieval quality decay — usually a bad ingestion or an embedding-model change. Fires *before* users complain. |
| `rag_loop_exhausted_total` **rises** | The agent is failing to find context and burning money doing it |
| `cost_usd_total` per conversation **rises** | Runaway loop or a prompt-size regression |

> Alerting on **retrieval score distribution** rather than only errors is a genuinely senior instinct. Bad RAG doesn't throw exceptions — it returns confident nonsense with a 200.

### 3. Logs

Structured JSON, one line per request, carrying `trace_id`, `tenant_id`, `route`, `latency_ms`, `cost_usd`, `citation_count`. Never log document content or PII — log ids and hashes.

### The feedback loop

```
thumbs-down → join to trace + citations → weekly triage
    ├─ retrieval miss   → fix chunking / add metadata / add to golden set
    ├─ generation issue → fix prompt → re-run evals → ship
    ├─ missing document → ingest it
    └─ genuinely unanswerable → confirm it abstained (that's a pass, not a fail)
```

---

## What went wrong, and what fixed it

**Prepare two of these properly.** This section wins interviews — it is the only part that cannot be faked from a tutorial.

### 1. Retrieval returned the right *topic* and the wrong *quarter*

**Symptom.** "How did the Q3 fintech LinkedIn campaign do?" returned the Q1 brief. Semantically near-identical documents; the embedding barely encodes the quarter.

**Root cause.** Dates were prose inside the chunk. Cosine similarity does not understand recency.

**Fix.** Extract quarter/channel/segment/date at ingestion into **metadata**, parse them out of the query, and apply as a Pinecone **pre-filter**. Turn a semantic problem into a filtering problem.

**Lesson to state:** *"Anything you'd put in a SQL `WHERE` clause should be metadata, not prose in the embedding."*

### 2. Confident arithmetic that was wrong

**Symptom.** Asked for ROAS across channels, it produced a fluent, plausible, wrong number — by retrieving chunks that *mentioned* ROAS and doing mental arithmetic.

**Root cause.** Numeric aggregation routed to a vector store. A category error.

**Fix.** The router: anything comparative, aggregate or arithmetic goes to SQL. Numbers come from the database or they don't come at all. The generated SQL is shown alongside the answer.

**Lesson:** *"Vector search retrieves; it doesn't compute. Once I separated those, a class of errors disappeared instead of being prompt-engineered around."*

### 3. The self-correction loop became an infinite loop

**Symptom.** On unanswerable questions the grader rejected context, the rewriter reformulated, retrieval failed again — forever. One question burned ‹$X› before anyone noticed.

**Root cause.** No bound, and no defined behaviour for "the answer isn't in the corpus."

**Fix.** `attempts` capped at 2; on exhaustion return an explicit *"I don't have information about X"* with what *was* found. Added `rag_loop_exhausted_total` and alerted on it. Added the unanswerable slice to the golden set so abstention is a tested behaviour.

**Lesson:** *"Every agent loop needs a bound and a defined behaviour at the bound. 'I don't know' is a feature you have to build."*

### 4. Duplicate chunks quietly degraded quality

**Symptom.** Answers grew repetitive; the context window filled with three copies of the same paragraph.

**Root cause.** Re-uploading an edited document appended new chunks without removing the old ones.

**Fix.** `content_hash` in the document registry; on re-ingest, delete the old chunks by `document_id` filter, then upsert. Idempotent ingestion.

**Lesson:** *"Ingestion is a data pipeline and needs pipeline discipline — idempotency, dedup, lineage. Retrieval bugs are usually ingestion bugs."*

### 5. Multi-turn broke on pronouns

**Symptom.** "Draft a revised one" retrieved nothing — embedding a pronoun retrieves noise.

**Fix.** A query-rewriting step that resolves the question against history into a standalone query before retrieval.

---

## Cost and latency

Have numbers, and have levers:

| Lever | Effect |
|---|---|
| **Route cheaply.** Small model for routing/grading, large only for synthesis | Most of the cost saving |
| **Semantic cache** on common questions | Marketing teams ask the same ‹20› questions |
| **Prompt caching** for the static system prompt and schema | |
| **Rerank instead of stuffing.** 5 good chunks beat 20 mediocre ones | Cheaper *and* more accurate |
| **Cap `max_tokens`** per node | Bounds the worst case |
| **Stream** | Halves *perceived* latency without touching real latency |

Latency budget for a hybrid query (illustrative): route ‹0.3s› → retrieve ‹0.4s› → rerank ‹0.3s› → SQL ‹0.5s› → synthesise ‹3–6s streamed›. Time-to-first-token is what users feel.

---

## Security, tenancy and compliance

- **Tenant isolation** — Pinecone namespace per tenant *and* `tenant_id` metadata filter. Belt and braces, because a namespace bug shouldn't leak data.
- **Document-level access** — `access` metadata on chunks; the retriever filters by the caller's clearance. Restricted briefs are invisible, not merely unmentioned.
- **Read-only database role** for the SQL tool, granted on reporting tables only. Enforced by Postgres, not by a prompt.
- **Prompt injection** — a retrieved document could contain "ignore your instructions." Mitigations: retrieved content is delimited and labelled untrusted data, the SQL tool validates before executing, and the compliance node runs on output. Be honest: *"input filtering reduces volume; it isn't a boundary. The real controls are least privilege and output checking."*
- **PII** — logs carry ids, never content.

---

## How to explain this in an interview

### The structure that works

1. **Problem (30s)** — three silos, one question that crosses all three.
2. **Shape (30s)** — FastAPI → LangGraph router → Pinecone / SQL / compliance.
3. **One hard thing (60s)** — pick *one*: the router, the self-correction bound, or the eval harness. Go deep.
4. **Results (20s)** — one engineering metric, one business metric.
5. **Hand over a thread** — "the part I found hardest was X, happy to go into it."

### Things to say that signal seniority

- *"I made the routing smart so most queries could stay dumb."*
- *"I evaluate retrieval and generation separately, because when quality drops I need to know which half broke."*
- *"Every agent loop has a bound and a defined behaviour at the bound."*
- *"Anything you'd put in a SQL `WHERE` clause belongs in metadata, not in the embedding."*
- *"Vector search retrieves; it doesn't compute."*
- *"Bad RAG doesn't throw exceptions. It returns confident nonsense with a 200, which is why I alert on retrieval score distribution."*
- *"The read-only grant is in Postgres, not in the prompt."*

### Traps

| Trap | What to do |
|---|---|
| "Why not just a bigger context window?" | Cost scales with context; attention degrades in the middle of long contexts; you still need access control and citations. Retrieval is also *cheaper* per query. |
| "Isn't this just a wrapper?" | Agree the LLM call is the easy part. The work is ingestion, routing, evaluation, guardrails and observability — name the failure modes you fixed. |
| "How do you know it works?" | Golden set, split metrics, CI regression gate, online feedback. This is why the eval section exists. |
| "Why LangGraph and not just Python?" | Fair question. Answer: checkpointing, streaming events, and a graph you can test node by node. And say you'd use a plain function for a fixed chain — you did, for ‹70%› of traffic. |
| Being asked for a number you don't have | *"I tracked it but don't remember the exact figure — the shape was X."* Never invent. |

### Draw this on the whiteboard

Boxes: **Client → FastAPI → Router → {Retrieve+Rerank, SQL, Compliance} → Synthesise → Respond**, with **Pinecone** and **Postgres** underneath and **traces/metrics** to the side. Then say: *"the interesting part is this edge"* and point at the grader loop back into retrieval.

---

## Situational questions

**Q. A stakeholder says the bot gave a wrong number in a board deck. What do you do?**
Contain, diagnose, fix, prevent. Get the message id → pull the trace → determine whether it was routed to SQL at all (if not, a routing bug), whether the SQL was wrong (a generation bug), or whether it was answered from a stale document (a retrieval/freshness bug). Fix the class, not the instance. Add the question to the golden set so it can never regress. Then the process question: if the bot's numbers reach board decks, answers need the generated SQL attached and a "verify before external use" affordance.

**Q. Legal says the assistant produced a non-compliant claim.**
Treat as a sev incident. Trace it: did the compliance node run, did it retrieve the right rules, did it pass something it should have caught? Most likely the rule chunk wasn't retrieved. Short term: tighten the compliance retrieval (rules are few — retrieve *all* applicable ones rather than top-k). Long term: this is why the node escalates rather than silently revising. And be clear with legal about what the guardrail is and isn't.

**Q. Users say "it's slow" but your p50 is fine.**
p50 isn't the experience — p95 is, and time-to-first-token more so. Check whether the slow ones are all one route (hybrid queries doing SQL and retrieval serially — parallelise them). Check whether step-level progress is rendering. Often "slow" means "silent."

**Q. Adoption stalled after launch. Why, and what do you do?**
Look at the data before theorising: are people asking questions and getting bad answers (quality), or not asking at all (discovery/trust)? Thumbs-down rate and question volume separate those. If it's trust, citations and showing the SQL matter more than model quality. If it's discovery, meet users where they are — the Slack surface beat the web UI for exactly this reason.

**Q. Your PM wants an agent that also *sends* campaigns.**
Push back on scope, not ambition. Read-only actions and write actions are different risk classes. If we do it: human-in-the-loop confirmation, a dry-run mode, an audit log, scoped credentials, and a rollback. I'd want the read-only system's task success rate to be solid before wiring anything that spends money.

---

## Scenario questions

**Q. The corpus grows from 200 documents to 50,000. What breaks first?**
Retrieval precision — more near-duplicates competing. Mitigations in order: harder metadata filtering, rerank becomes essential rather than an optimisation, consider splitting namespaces by doc_type, revisit chunk size. Cost and latency of ingestion also become real, so it needs incremental indexing. What *doesn't* break is Pinecone's query latency — that's the point of using it.

**Q. Marketing wants this for five customer brands, each with different voices and rules.**
Already handled by tenancy: namespace per brand, metadata filter, brand voice and compliance rules ingested per tenant. The prompt loads the brand's voice guide as context rather than being fine-tuned per brand. The thing to watch is the shared eval set — each brand needs its own golden slice.

**Q. Your embedding provider releases a much better model.**
Not a drop-in: dimensions differ, so it's a new index. Plan: build the new index in parallel, run the golden set against both, compare Recall@k and MRR, then cut over with a feature flag and keep the old index for rollback. Mention that all similarity thresholds are model-specific and must be re-tuned — scores are not comparable across models.

**Q. Latency doubles overnight with no deploy.**
No deploy means it's upstream or data. Check provider status and your own latency-by-stage histogram — that's why the metric is labelled by stage. If it's `retrieve`, suspect index size or a noisy neighbour; if `generate`, suspect the provider or a context-size regression from an ingestion that made chunks bigger.

**Q. How would you add multi-modal — campaign creatives, images?**
Two options, and I'd be explicit about the trade: caption images with a vision model at ingestion and embed the captions (cheap, works with existing text infrastructure, loses detail), or use a multimodal embedding model with a separate index (better fidelity, more infrastructure). I'd start with captions and measure before building the second.

---

## Technical questions

### RAG

**Chunk size trade-off?** Small chunks → precise retrieval, missing context. Large → more context, diluted embeddings and wasted tokens. Structure-aware beats a fixed number when documents have real structure. Measure it: chunk size is a hyperparameter, and I tuned it against Recall@k on the golden set.

**Overlap — why?** So a fact spanning a boundary isn't cut in half. Cost is duplicate content and a larger index.

**Hybrid search — why not dense only?** Dense retrieval is weak on exact tokens: campaign codes, product names, acronyms. BM25 nails those. Fuse with Reciprocal Rank Fusion.

**What is RRF?** Combines rankings by summing `1/(k + rank)` across retrievers. Rank-based, so it doesn't need scores from different systems to be on a comparable scale — which they never are.

**Bi-encoder vs cross-encoder?** Bi-encoder embeds query and document separately (fast, precomputable, less accurate). Cross-encoder scores the pair jointly (accurate, can't precompute). Hence retrieve-then-rerank.

**HNSW, roughly?** A navigable small-world graph in layers; search descends from sparse upper layers to dense lower ones. Approximate — you trade a little recall for a lot of speed, tuned by `ef_search`.

**Cosine vs dot product?** Cosine normalises out magnitude; dot product doesn't, so longer vectors score higher. If the model returns normalised vectors they rank identically. Know which your model does.

**"Lost in the middle"?** Models attend less well to the middle of a long context. So: fewer, better chunks, and put the most relevant at the edges.

**How do you stop hallucination?** You reduce it; you don't stop it. Ground with retrieval, instruct to answer only from context, require citations, grade faithfulness in eval, and let it abstain. Abstention has to be a designed, tested behaviour.

### LangGraph / agents

**Why a graph over a ReAct loop?** Explicit control flow. I can test each node, resume from a checkpoint, stream node-level progress, and reason about what happens on failure. A free-form ReAct loop is harder to bound and harder to debug.

**How is state managed?** A typed state dict; nodes return partial updates that are merged. `add_messages` appends rather than replaces. Checkpointers persist state per `thread_id`, which is what makes conversations resumable.

**Conditional edges?** A function reads state and returns the next node name — that's how routing and the grader retry are expressed.

**How do you prevent infinite loops?** A counter in state, a recursion limit on the graph, and a defined terminal behaviour. All three.

**Multi-agent vs single agent with tools?** Start with one agent and tools. Multiple agents when responsibilities genuinely differ and each needs its own prompt and tool set — and accept the cost: more latency, more failure surface, harder debugging.

### FastAPI

**Why async here?** IO-bound workload. `async` lets one worker hold many in-flight requests cheaply.

**What breaks it?** A synchronous blocking call inside an async route blocks the event loop for everyone. CPU-bound work goes to `run_in_threadpool` or a worker.

**Streaming?** SSE via `EventSourceResponse`, driven by `astream_events`. Also stream *step* events — perceived latency is the product.

**Background tasks?** Ingestion returns a job id immediately and indexes out of band. For anything that must survive a restart, a real queue rather than `BackgroundTasks`.

**Dependency injection?** `Depends` for auth, DB session, tenant resolution — and it makes tests easy via dependency overrides.

### Pinecone

**Namespace vs metadata filter?** Namespaces are a hard partition and are queried separately — good for tenants. Metadata filters are applied within a query — good for facets like doc_type or quarter. I use both.

**Why does the index have a fixed dimension?** It's built for one embedding model's output. Changing model means a new index.

**Upsert semantics?** Deterministic chunk ids (`{document_id}:{chunk_index}`) so re-ingestion overwrites rather than duplicates. Deleting a document is a delete-by-filter on `document_id`.

### SQL / text-to-SQL

**How do you keep it safe?** Read-only role with minimal grants, single-`SELECT` validation before execution, injected `LIMIT`, statement timeout. Prompt instructions are the last line of defence, not the first.

**How do you evaluate it?** Execution accuracy — run both queries, compare result sets. String comparison would fail on correct-but-different queries.

**Schema too big for the prompt?** Retrieve the relevant tables first — RAG over the schema — and pass only those, with column descriptions and a few worked examples.

### Evaluation & monitoring

**Most important RAG metric?** Recall@k, because it caps everything downstream. Then faithfulness, because that's the trust-destroying failure.

**LLM-as-judge — is it trustworthy?** Usefully, not fully. Biased toward length and toward its own style. Control with a fixed rubric, a held-out human-labelled subset to validate the judge, and position-swapping in pairwise comparisons.

**How do you catch quality regressions before users?** Golden set in CI on every PR that touches prompts, chunking or the graph, with a threshold gate; plus the retrieval-score alert in production.

**What would you monitor if you could only pick three?** Retrieval score distribution, loop-exhaustion rate, and cost per conversation. The first catches silent quality decay, the second catches the agent failing, the third catches runaway spend.

---

## Sixty-second answers to have ready

| Question | The shape of your answer |
|---|---|
| "Walk me through the project." | The 90-second pitch. |
| "Hardest technical problem?" | The self-correction loop that didn't terminate — symptom, root cause, fix, lesson. |
| "How do you know it's good?" | Split metrics, golden set, CI gate, online feedback. |
| "What would you do differently?" | Build the eval harness *first*. I built it after the third quality complaint, and every fix before that was guesswork. |
| "What would you build next?" | Scheduled proactive insights ("this campaign is tracking 30% below benchmark") — the shift from pull to push. |
| "What are its limits?" | It can't do genuine causal attribution; it can't replace legal review; it degrades when the corpus is stale, so ingestion freshness is an operational commitment, not a one-off. |

> **The best closing line in an interview:** *"The thing I'd change is that I built the evaluation harness after the third quality complaint instead of before the first. Everything I fixed before that, I fixed by guessing."*
>
> It's honest, it's specific, and it demonstrates exactly the judgement they're hiring for.
