# Marketing Copilot

**RAG and an agent over a marketing team's documents, campaign numbers and compliance rules —
with the routing, the evaluation and the monitoring shown rather than described.**

---

## The problem

A marketing team's knowledge sits in three places that cannot talk to each other.

| Where | What's in it | Who can use it |
|---|---|---|
| **Documents** | Brand voice, campaign briefs, personas, battlecards, post-campaign retros | Anyone in theory; nobody in practice, because search is keyword-based |
| **A database** | Spend, impressions, clicks, conversions, pipeline by channel/segment/quarter | Analysts. Everyone else files a ticket |
| **A compliance PDF** | Which claims are allowed, and what each one needs | Legal. Marketers meet the rules when their copy is rejected |

## The question that decides the architecture

> *"Why did the Q3 LinkedIn fintech campaign underperform, and draft a revised brief."*

One sentence needing **numbers** (SQL), **context** (retrieval), **synthesis**, **generation**, and
**verification** — and the number and order of those steps depends on the question. That is the
honest reason to reach for an agent rather than a chain.

**And why most of it is not agentic.** *"What is our brand voice"* is one retrieval and one
generation. *"What did we spend on LinkedIn"* is one tool call. The **routing** is the intelligent
part so the common paths stay cheap: an agent loop is latency, cost and a new way to fail, and it
is worth paying for only where it earns its place.

---

## The graph

```
  route ──► retrieve ──► grade ──► rewrite ──┐
        ├─► sql          │        ▲──────────┘  bounded at 2 attempts
        ├─► hybrid ──────┘
        └─► generate
                 └──► synthesise ──► compliance ──► answer
```

| Node | What it does | The decision worth knowing |
|---|---|---|
| `route` | Classifies into rag / sql / hybrid / generate | Temperature 0, one word out. A classifier that varies between identical inputs cannot be evaluated. Ambiguous → `hybrid`: slower beats wrong |
| `retrieve` | Metadata pre-filter, then hybrid search, fused by RRF | Quarter/channel/segment are **filters**, not embedding hopes |
| `grade` | Judges whether the context can answer the question | Grades the *context*, not the answer — catching a retrieval miss before a fluent answer is written from it |
| `sql` | Text-to-SQL, validated, executed read-only | The database refuses writes; the prompt is the last line of defence, not the first |
| `synthesise` | Writes from what was gathered, or abstains | Abstention is asked for explicitly. Left alone, a model always finds something to say |
| `compliance` | Pattern-checks generated assets against every rule | Escalates rather than silently rewriting |

---

## Four decisions that carry the project

**Metadata is filtered, not embedded.** Two campaign briefs are semantically near-identical and
differ by a quarter the embedding barely encodes, so asking about Q3 returned the Q1 brief. Anything
that would be a SQL `WHERE` clause is extracted at ingestion and applied before similarity is
consulted.

**Numbers never enter the vector store.** Vector search retrieves; it does not compute. Asked for
ROAS, a retrieval-only system finds chunks *mentioning* ROAS and does confident mental arithmetic —
which is how a fluent wrong number reaches a board deck. The generated SQL is returned with the
answer so it can be checked.

**Chunking follows structure, not length.** A brief's headings are its semantics: the "Audience"
section of one campaign must never merge with the "Budget" of the next, which is exactly what a
fixed character window does. Long sections fall back to a windowed split, so the strategy degrades
rather than breaks.

**The loop is bounded.** Two retrieval attempts, then stop, with a defined behaviour at the bound:
say what was searched and what was not found. An unbounded self-correction loop is a bill and an
outage.

---

## Evaluation

The principle the harness is built on: **score retrieval and generation separately.** When quality
drops you have to know which half broke — one end-to-end number cannot tell you.

| Family | Metrics |
|---|---|
| **Retrieval** | Recall@k *(the ceiling on everything downstream)*, MRR |
| **Generation** | Answer quality, grounded rate |
| **Agent** | Routing accuracy, mean attempts, abstention correctness |
| **End to end** | Pass rate — answered well **and** grounded |

The golden set declares, per case, the route it should take and the documents it should find. Its
**unanswerable** slice is the important one: those questions have no answer in the corpus and the
pass condition is that the copilot refuses them. A system that never says "I don't know" is not
accurate, only confident.

Runs are stored, not printed. A single score tells you nothing; a score next to last week's tells
you whether the change you shipped helped.

---

## Monitoring

Bad RAG **does not throw exceptions**. It returns confident nonsense with a 200, so error rate — the
metric most services are watched on — is exactly the one that will not catch it.

| Watch | Because |
|---|---|
| `avg_top_score` falling | Retrieval decay. Fires before users complain |
| `loop_exhausted` rising | The agent is failing and spending money doing it |
| `p95_latency` | The median is not the experience; the slow requests are |

Every answer is stored with its route, citations, attempts, latency and retrieval score, in the same
database as the campaign data. So *"show me every thumbs-down answer and the chunks it cited"* is one
join — and that query is how retrieval problems are actually found.

---

## Layout

```
database.py    schema — business tables and observability tables together, deliberately
corpus.py      the seeded demo workspace: documents, campaign rows, rules, golden set
retrieval.py   structure-aware chunking, embeddings, hybrid search, RRF, filters
tools.py       text-to-SQL with guardrails; the compliance checker
graph.py       the LangGraph agent
evaluation.py  the harness and the metric glossary
providers.py   OpenAI / Google — chat and embeddings from one key
content.py     the written explanation served to the UI
router.py      every endpoint
```

## Configuration

The user brings their own model key through the UI; nothing here holds a platform model credential.

| Setting | Effect |
|---|---|
| `MARKETINGCOPILOT_PINECONE_INDEX` | Use Pinecone. Unset → brute-force cosine over SQLite, which is exact and fine at this corpus size |
| `PINECONE_API_KEY` | Shared platform account (in `core.config`) |

One index per project: an index has a fixed dimension and cannot hold vectors from two embedding
models. Changing the embedding model means a new index and a re-index.

## API

All paths are mounted under `/marketingcopilot` and require the platform auth cookie.

| Method | Path | Returns |
|---|---|---|
| `GET` | `/overview` | The problem, the stack, corpus counts |
| `GET` | `/explain` | The graph node by node, the failure modes, the metric glossary |
| `GET` `POST` | `/setup` | The user's provider, models and key |
| `POST` | `/index` | Chunk and embed the corpus |
| `GET` | `/corpus`, `/corpus/{id}` | Documents, campaign rows, rules; one document with its chunks |
| `GET` | `/suggestions` | Questions that each take a different route |
| `POST` | `/chat` | Run the agent; returns the answer with its route, steps, citations and SQL |
| `GET` | `/conversations`, `/conversations/{id}` | History |
| `POST` | `/feedback` | Rate an answer |
| `POST` | `/compliance/check` | Check arbitrary copy against the rules |
| `GET` | `/eval/dataset`, `/eval/runs`, `/eval/runs/{id}` | The golden set and past runs |
| `POST` | `/eval/run` | Run the evaluation |
| `GET` | `/monitor` | The dashboard |

## What this does not claim

- **The compliance node is a guardrail, not a gate.** It reduces what reaches a human reviewer. It
  cannot make a claim legally safe, and a rephrasing in ordinary words will pass it — which is the
  honest limit of pattern matching.
- **It does not do causal attribution.** On "why did this underperform" it reports what the retro
  said and what the numbers show. Correlation with a documented explanation is not proof.
- **The corpus is fictional**, and freshness is an operational commitment: a stale corpus produces
  confident, out-of-date answers, which is the failure mode this design is least able to detect.
