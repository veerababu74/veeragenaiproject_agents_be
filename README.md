# veeragenaiproject_agents_be

Agent backends for the Veera AI workspace. **One FastAPI app, one deployment, one `.env`** — with
each project as a self-contained module mounted under its own URL prefix.

```
veeragenaiproject_agents_be/
├── main.py                 the app — mounts everything in the registry, names no project
├── requirements.txt        shared dependencies
├── .env / .env.example     one file: shared values, plus <SLUG>_ prefixed per-project overrides
│
├── core/                   written once, used by every project
│   ├── auth.py             verifies the platform's JWT cookie
│   ├── config.py           shared settings + project_value() for per-project lookups
│   ├── database.py         per-project SQLite, plus the retention sweep
│   ├── embeddings.py       Gemini embeddings (768-dim)
│   ├── storage.py          Hugging Face bucket, one prefix per project
│   ├── vectors.py          Pinecone, one index per project
│   └── registry.py         the Project dataclass
│
└── projects/
    ├── __init__.py         REGISTRY — the single list main.py reads
    ├── simpleagent/        mounted at /simpleagent
    └── insidellm/          mounted at /insidellm
```

Every route is namespaced by project: `/simpleagent/agent`, `/simpleagent/run/stream`, and so on.
`GET /health` lists what is mounted.

## Adding a project

Nothing in `core/` or `main.py` changes. Four steps:

**1. Create `projects/<slug>/`** with your routers and services. Import shared pieces from `core`:

```python
from core.auth import current_user_id          # same signed-in user as the workspace
from core.database import Database             # your own SQLite file
from core.storage import Bucket                # your own prefix in the shared bucket
from core.vectors import VectorStore           # your own Pinecone index
```

**2. Declare the project's storage** in `projects/<slug>/resources.py`, so its index and prefix are
stated in one place and overridable from the environment:

```python
from core.config import project_value
from core.embeddings import EMBEDDING_DIMENSION
from core.storage import Bucket
from core.vectors import VectorStore

SLUG = "myproject"
bucket = Bucket(project_value(SLUG, "storage_prefix", SLUG))
vectors = VectorStore(project_value(SLUG, "pinecone_index", f"{SLUG}-rag"), EMBEDDING_DIMENSION)
```

**3. Expose a `PROJECT`** in `projects/<slug>/__init__.py`:

```python
PROJECT = Project(
    slug="myproject",
    title="My Project",
    description="...",
    routers=(foo.router, bar.router),
    initialize=init_db,           # runs once at startup
    cleanup=cleanup_expired_data, # runs on the retention schedule
)
```

**4. Add it to `REGISTRY`** in `projects/__init__.py`. That is the only shared file you touch.

Then add any new dependencies to the root `requirements.txt`, and — if the project needs its own
index or prefix — a `MYPROJECT_PINECONE_INDEX` line to `.env`.

## Configuration

| Scope | How it is read | Example |
| --- | --- | --- |
| Shared | A field on `Settings` in `core/config.py` | `JWT_SECRET`, `PINECONE_API_KEY`, `RETENTION_HOURS` |
| Per project | `project_value(slug, key, default)` → `<SLUG>_<KEY>` | `SIMPLEAGENT_PINECONE_INDEX` |

Per-project values are looked up dynamically, so a new project's settings never require editing
`core/config.py`.

## How this fits the platform

| Concern | How it works |
| --- | --- |
| Authentication | No project issues tokens. `core/auth.py` verifies the HS256 cookie set by `veeragenai_projects_be` using the same `JWT_SECRET`, so a signed-in workspace user is signed in here with the same user id |
| Frontend | `veeragenai_projects_fe`, under `src/features/projects/<project>/`. `src/lib/agentsApi.js` is a factory — `createAgentsApi('<slug>')` — pointed at one shared host |
| Production routing | A single rewrite in the frontend's `vercel.json`: `/agents-api/:path*` → this deployment. Every project is reached at `/agents-api/<slug>/…` |
| Catalogue and blog | Seeded from `veeragenai_projects_be` (`Admin/landing.py`, `Blog/project_guides.py`) |
| Storage | One Hugging Face bucket with a per-project prefix; a dedicated Pinecone index per project, since an index has a fixed dimension |
| Databases | One SQLite file per project in `./data` (or `DATA_DIR`) |
| Retention | `RETENTION_HOURS` (default 48). Each project's `cleanup` runs hourly; one failing sweep never stops the others |

## Running locally

```bash
python -m venv .venv && .venv/Scripts/activate     # source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env                                # fill in JWT_SECRET and the storage keys
python main.py                                      # http://localhost:8004
```

`JWT_SECRET` must match `veeragenai_projects_be`, otherwise every authenticated call returns 401.

## Deployment

One deployment for all projects, from the repository root:

```bash
fastapi deploy
```

Then point the frontend's `/agents-api` rewrite at the assigned URL.

## Projects

### [simpleagent](./projects/simpleagent) — mounted at `/simpleagent`

Build one agent, attach up to ten tools, and watch it decide in real time: what it was told, what it
reasoned, which tool it picked, what came back, and in what order. Five model providers, twelve
built-in tools plus user-defined HTTP tools, document retrieval with four chunking strategies.

Full write-up: [projects/simpleagent/README.md](./projects/simpleagent/README.md).

### [insidellm](./projects/insidellm) — mounted at `/insidellm`

A visual walkthrough of a real GPT-2 forward pass: tokenization with every byte-pair merge, real
embeddings, all 144 attention matrices, one head's arithmetic in full, and the logit lens showing the
prediction sharpen layer by layer.

It holds **no model at runtime**. GPT-2 is implemented in numpy and run at build time over the real
pretrained weights; the captured intermediates are committed as 648 KB of JSON and the server just
serves the bytes. That is what lets it run on a 2 CPU / 2 GB host — and it registers neither an
`initialize` nor a `cleanup` hook, being read-only.

Full write-up: [projects/insidellm/README.md](./projects/insidellm/README.md).
