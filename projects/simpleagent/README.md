# SimpleAgent

One agent, up to ten tools, and a live view of how it decides.

When an agent answers you, you normally see the answer and nothing else. You cannot tell whether it
searched or guessed, whether it did the arithmetic or estimated it, or whether it called one tool or
four. SimpleAgent exists to remove that opacity: you build a single agent, attach the tools it may
use, and every decision it makes is streamed to the browser while it happens.

---

## Contents

- [The core idea: rounds and tool order](#the-core-idea-rounds-and-tool-order)
- [Architecture](#architecture)
- [Request lifecycle](#request-lifecycle)
- [The trace](#the-trace)
- [Tools](#tools)
- [Documents, chunking and retrieval](#documents-chunking-and-retrieval)
- [Authentication](#authentication)
- [Data model](#data-model)
- [Retention](#retention)
- [API reference](#api-reference)
- [Running locally](#running-locally)
- [Deployment](#deployment)
- [Design decisions](#design-decisions)

---

## The core idea: rounds and tool order

An agent does not plan a whole run up front. It is called, it either answers or asks for tools, the
tools run, and it is called again with what they returned. Each pass of that cycle is a **round**.

```
Round 1   model: "I need today's date and the converted amount"
          -> tool 1  current_datetime
          -> tool 2  currency_convert
Round 2   model: "I have the total, now divide across the days"
          -> tool 3  calculator
Round 3   model: final answer
```

Two numbers make that readable, and the tracer exists to produce them:

| Number | Meaning | Why it matters |
| --- | --- | --- |
| `round` | Which pass of the think-act loop a step belongs to | A run that reaches round 2 is one where the model *reacted* to a tool result instead of planning everything up front |
| `tool_order` | The position of a tool call across the whole run | Makes "searched, then opened the page it found, then did the arithmetic" legible as a sequence |

Tools sharing a round were chosen **simultaneously** by one model call. A tool in a later round was
chosen **after reading** an earlier result. That distinction is the whole point of the display.

---

## Architecture

This project is one module inside the shared agents service, mounted at **`/simpleagent`**. Auth,
configuration, SQLite handling, embeddings, object storage and the vector store all come from `core/`
at the repository root; everything below is what this project adds on top.

```
                         shared auth cookie
  React console  ──────────────────────────────►  /simpleagent/…
   │  live trace                                   │
   │                                               ├── core/auth.py         verifies the platform's JWT
   │                                               ├── resources.py         this project's bucket prefix + Pinecone index
   │                                               ├── database.py          schema + retention sweep
   │                                               ├── routers/             agent, tools, documents, runs, examples
   │                                               └── services/
   │                                                     ├── runner.py        LangGraph think-act loop
   │                                                     ├── tracing.py       callback that records every step
   │                                                     ├── tool_builder.py  assembles the agent's tools
   │                                                     ├── builtin_tools.py the catalogue
   │                                                     ├── chunking.py      extraction + 4 strategies
   │                                                     └── llm_provider.py  the five providers
   │                                                          │
   └────────  server-sent events  ◄────────────────────────── ┤
                                                              ├── SQLite      data/simpleagent.db (48h)
                                                              ├── Pinecone    simpleagent-rag, namespaced per user
                                                              └── Hugging Face originals under simpleagent/
```

Everything the agent needs is resolved per request from SQLite; nothing is held in process memory
between requests, so the service is safe to run with more than one worker.

---

## Request lifecycle

`POST /simpleagent/run/stream` returns `text/event-stream`. Events arrive in this order:

| Event | Payload | When |
| --- | --- | --- |
| `start` | `run_id` | Immediately, before any model call |
| `step` | a trace step (see below) | Every decision, tool call and tool result |
| `token` | `text` | Each chunk of the model's reply as it streams |
| `done` | `answer`, `rounds`, `tool_calls`, `tokens`, `duration_ms` | Once, at the end |
| `error` | `error` | Instead of `done` if the run failed |

Internally, `services/runner.py` compiles a two-node LangGraph — a model node and a `ToolNode` — with
a conditional edge that loops back while the model keeps requesting tools:

```python
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))
graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")     # the loop that makes rounds possible
```

The run is driven with `graph.astream_events(...)` and a `RunTracer` attached as a callback. A queue
bridges the two: the tracer pushes each step onto it as it is recorded, and the SSE generator drains
the queue between graph events, so the browser sees a step the moment it happens rather than at the
end of the round.

---

## The trace

`services/tracing.py` is a LangChain `AsyncCallbackHandler`. Callbacks are used rather than manual
logging because they fire from inside the loop and therefore observe the real order of events, not
the order somebody remembered to log them in.

| Step type | Recorded from | Content |
| --- | --- | --- |
| `context` | Written before the run starts | The exact system message, plus the name, description and argument schema of every tool the model may pick from |
| `question` | The request | The user's text |
| `think` | `on_llm_end`, when the message carries tool calls | What the model said while deciding, which tools it chose, and the arguments it chose them with |
| `tool_call` | `on_tool_start` | Tool name, arguments, and its `tool_order` |
| `tool_result` | `on_tool_end` | The returned text and the call's duration |
| `tool_error` | `on_tool_error` | The failure, in the position where it happened |
| `answer` | End of the run | The final text, with round count, tool count and token usage |

Two details worth knowing:

- **A model message with no tool calls is not recorded as `think`.** That message *is* the answer, and
  recording it twice would show the same text in two places.
- **The `context` step is written before the model is ever called.** A tool choice can only be judged
  against the information the choice was made from, so the prompt and tool descriptions are captured
  first and shown next to the decision they produced.

Steps stream live and are written to SQLite in a single `executemany` at the end of the run: the
browser needs each event immediately, but the database wants one transaction so the sequence stays
contiguous.

---

## Tools

Twelve built-ins ship in `services/builtin_tools.py`. Six need no credential at all, which is
deliberate — watching an agent sequence tools should not require signing up for a search provider.

| Tool | Type | Key |
| --- | --- | --- |
| Date & Time | `datetime` | — |
| Calculator | `calculator` | — |
| Web Search (DuckDuckGo) | `web_search` | — |
| Wikipedia | `wikipedia` | — |
| Currency Converter | `currency` | — |
| Web Page Reader | `web_fetch` | — |
| HTTP Request | `http_request` | — |
| Tavily Search | `tavily` | required |
| Google Search (Serper) | `google_search` | required |
| Slack | `slack` | required |
| GitHub | `github` | required |
| Document Search | `document_search` | uses your Gemini key |

Every tool is a plain REST call made with `aiohttp` rather than a provider SDK, so adding one never
adds a dependency. Failures are **returned as text, not raised** — a tool that throws would end the
run, when the useful behaviour is for the agent to notice the failure and try something else.

The calculator evaluates an AST directly and only reaches numeric literals and a fixed operator
table, so no name lookup, attribute access or function call can be expressed in its input.

### Custom tools

A custom tool is described once as a URL, a method and a list of typed fields. The fields become the
arguments the model fills in; a `{placeholder}` in the URL is substituted from them and then removed
so the same value is not also sent as a query parameter. Building the argument model uses
`pydantic.create_model` rather than `type()` — the `name: (type, FieldInfo)` mapping only means
anything to `create_model`, and passing it to `type()` leaves the fields unannotated, which pydantic
v2 rejects.

A tool that fails to build is skipped and logged rather than taking down the agent it belongs to.

---

## Documents, chunking and retrieval

Extraction is **structural, then flattened**. PDFs, DOCX files and CSVs are read into typed blocks —
`heading`, `paragraph`, `bullet`, `table`, `image` — and only then joined into text. Three strategies
work on the flat text, but context-aware needs to know what each piece of text *was*, and that cannot
be recovered from a flat string afterwards.

| Strategy | How it splits |
| --- | --- |
| `fixed` | A plain character window with overlap |
| `recursive` | Paragraph, then sentence, then word boundaries (LangChain's splitter) |
| `semantic` | Embeds each sentence and starts a new chunk where cosine similarity to the previous sentence drops below 0.72 |
| `context_aware` | A heading starts a chunk and is repeated at the top of every chunk beneath it; a table is never split except on row boundaries; bullets stay with their neighbours |

Accepted types are PDF, TXT, DOCX and CSV, with a **5 MB total allowance per user** across all their
documents.

The pipeline is synchronous on purpose. Extraction, chunking, embedding and indexing must all succeed
for a document to be searchable, and hiding that behind a job queue would only move the failure
somewhere the user cannot see it. If indexing fails after the file is stored, the stored copy is
deleted again so a half-ingested document never lingers.

Embeddings are Gemini, pinned to **768 dimensions** via `outputDimensionality` — the two offered
models have different native sizes and one Pinecone index cannot mix dimensions. Each user's vectors
live in their own Pinecone namespace, so a search can never reach another user's documents. The
search tool embeds queries with the model the user's documents were actually built with; defaulting
would silently mismatch and degrade every result.

---

## Authentication

Handled by `core/auth.py`, shared with every project in this service. No tokens are issued here: the
same HS256 cookie that `veeragenai_projects_be` sets is verified with a `JWT_SECRET` that **must be
identical** in both. A user signed in to the workspace is therefore already signed in here, with the
same user id, and every query in this project is scoped by that id.

---

## Data model

SQLite at `data/simpleagent.db` from the repository root (or under `DATA_DIR`). Connection handling,
schema bootstrap and the retention sweep come from `core/database.py`; only the tables are this
project's.

| Table | Holds |
| --- | --- |
| `agents` | One row per user — `UNIQUE(user_id)` enforces the single-agent rule in the schema, not just the API |
| `provider_keys` | One key per provider per user |
| `tools` / `custom_tools` | The user's tool library, and the HTTP spec for custom ones |
| `tool_links` | Which tools are attached to the agent; capped at 10 |
| `documents` | Upload metadata, chunk strategy and status |
| `runs` | One row per execution, with rounds, tool calls, tokens and duration |
| `run_steps` | The trace: `seq`, `round`, `step_type`, `tool_name`, `tool_order`, content |
| `messages` | Conversation history, replayed as memory (last 12 turns) |

---

## Retention

Everything is deleted **48 hours** after it is created: the agent, its tools, provider API keys,
documents, conversations and traces. A sweep runs hourly and removes the Pinecone vectors and the
Hugging Face originals *before* the rows that point at them, so nothing is orphaned in an external
service that we can no longer address.

---

## API reference

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/providers` | The five providers with suggested models |
| `GET` `POST` `DELETE` | `/agent` | Read, create-or-update, delete the agent |
| `GET` `POST` | `/keys`, `/keys/{provider}` | Which providers have a key saved; save or delete one |
| `GET` | `/tools/catalog` | The built-in catalogue |
| `GET` | `/tools` | The user's tool library, with attachment state |
| `POST` | `/tools/builtin`, `/tools/custom` | Add a tool |
| `POST` `DELETE` | `/tools/{id}/attach` | Attach or detach (max 10) |
| `DELETE` | `/tools/{id}` | Delete a tool |
| `GET` | `/documents/options` | Strategies, embedding models, accepted types, quota |
| `GET` `POST` `DELETE` | `/documents`, `/documents/upload`, `/documents/{id}` | Document management |
| `POST` | `/run/stream` | Run the agent (server-sent events) |
| `GET` | `/runs`, `/runs/{id}` | Recent runs; one run with its full trace |
| `GET` `DELETE` | `/conversations/{id}` | Read or clear a thread |
| `GET` | `/examples` | The ready-made agents |
| `POST` | `/examples/{id}/apply` | Configure the agent and attach that example's keyless tools |

**Every path is mounted under `/simpleagent`** — the table lists them relative to that, so
`/agent` is served at `/simpleagent/agent`.

Every route except `/providers`, `/tools/catalog`, `/documents/options` and `/examples` requires the
auth cookie.

---

## Running locally

The whole service runs as one app from the repository root:

```bash
cd ..\..                                            # repository root
python -m venv .venv && .venv/Scripts/activate      # source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env                                # fill in JWT_SECRET and the storage keys
python main.py                                      # http://localhost:8004/simpleagent
```

`JWT_SECRET` must match `veeragenai_projects_be`, otherwise every authenticated call returns 401.

This project reads two optional overrides from the root `.env`: `SIMPLEAGENT_PINECONE_INDEX`
(default `simpleagent-rag`) and `SIMPLEAGENT_STORAGE_PREFIX` (default `simpleagent`).

The frontend reads `VITE_AGENTS_API_URL` in development, defaulting to `http://localhost:8004`, and
appends the project slug.

---

## Deployment

Deployed with every other project in this repository, as one app, from the **repository root**:

```bash
fastapi deploy
```

The frontend reaches it through a single shared rewrite in `veeragenai_projects_fe/vercel.json` —
one rewrite serves every project, since each is namespaced by its slug:

```json
{ "source": "/agents-api/:path*", "destination": "https://<your-deployment>.fastapicloud.dev/:path*" }
```

Update that destination to the URL FastAPI Cloud assigns on first deploy.

---

## Design decisions

| Decision | Why |
| --- | --- |
| One agent per user | The subject is how a single agent decides. A second agent adds a selection step and a way to end up watching the wrong one |
| Ten tools maximum | Past roughly ten tools model tool-choice degrades and the trace stops being readable; the limit protects both |
| Tracing via callbacks | Callbacks fire inside the loop and see the true order of events |
| Stream live, persist once | The browser needs events immediately; the database wants one transaction so `seq` stays contiguous |
| Tool failures return text | A raised exception ends the run; a returned error lets the agent recover, which is the more interesting behaviour to watch |
| Model names typed, not enumerated | Providers release and retire models constantly; a fixed list would block a new model until redeploy |
| Keys stored per provider, not per agent | The key is needed when the agent is created, but changing the model should not mean re-entering it |
| Synchronous ingestion | A background queue would hide ingestion failures from the person who can fix them |
