# Inside an LLM

A visual walkthrough of a **real GPT-2 forward pass**, component by component.

Every number this project displays — token IDs, embedding vectors, attention weights, dot products,
logits — was read out of an actual forward pass over GPT-2's published pretrained weights. Nothing is
simulated, and nothing is illustrative.

---

## Contents

- [The constraint that shaped everything](#the-constraint-that-shaped-everything)
- [Architecture](#architecture)
- [The build step](#the-build-step)
- [What is captured](#what-is-captured)
- [The nine components](#the-nine-components)
- [The ten examples](#the-ten-examples)
- [Correctness](#correctness)
- [API reference](#api-reference)
- [Rebuilding the data](#rebuilding-the-data)
- [Design decisions](#design-decisions)
- [What this project does not claim](#what-this-project-does-not-claim)

---

## The constraint that shaped everything

The target host has **2 CPUs and 2 GB of memory**. Loading GPT-2 at runtime is not an option: PyTorch
alone is a larger resident set than the whole budget, before any weights.

The project accepts **no typed input** — there are ten fixed examples. That turns out to be the
solution rather than a limitation, because it means every number can be computed **once, ahead of
time**, and saved.

| | Build time (a developer's machine) | Runtime (the 2 GB host) |
|---|---|---|
| Model weights | 548 MB, downloaded once | **never loaded** |
| Heavy dependencies | numpy, regex | **none** |
| Memory | ~600 MB during twelve forward passes | a few hundred KB of cached JSON |
| Work per request | ~2 seconds per sentence | a file read |
| Output | 648 KB of JSON | under 80 KB per example |

The teaching argument is the stronger one anyway. An arbitrary sentence usually demonstrates nothing
in particular; each of the ten was chosen because it makes one specific mechanism visible.

---

## Architecture

```
BUILD (run by hand, never on the server)
  gpt2/model.safetensors  548 MB ─┐
  gpt2/vocab.json + merges.txt   ─┤
  examples.py  (ten sentences)   ─┴─► precompute/build.py
                                          │  tokenizers.py    real BPE, every merge recorded
                                          │  gpt2_numpy.py    instrumented forward pass
                                          ▼
                                     data/*.json   648 KB
                                          │
RUNTIME                                   ▼
  router.py ──► reads the bytes, returns them unparsed
  content.py ─► static explanations (no data files)
```

The runtime half imports nothing from `precompute/`. That directory ships with the project because it
*is* part of the explanation — the numpy implementation is the clearest statement of what the model
does — but the server never touches it.

---

## The build step

`precompute/gpt2_numpy.py` implements GPT-2 in numpy, longhand:

```python
x = wte[ids] + wpe[positions]
for each of 12 blocks:
    h        = layer_norm(x, ln_1)
    q, k, v  = split(h @ c_attn.W + c_attn.b)      # Conv1D: [in, out], applied directly
    scores   = q @ k.T / sqrt(64)
    scores   = where(causal_mask, -inf, scores)
    x        = x + (softmax(scores) @ v) @ c_proj.W + c_proj.b
    h2       = layer_norm(x, ln_2)
    x        = x + gelu(h2 @ c_fc.W + b) @ c_proj.W + b
logits = layer_norm(x, ln_f) @ wte.T                # output weights tied to input embeddings
```

Written out rather than imported because **a library returns an answer and discards the working**, and
the working is the entire subject. Implementing it by hand made every intermediate available for free.

Weights are read with a ~20-line safetensors reader (8-byte header length, JSON header, raw tensors),
memory-mapped so the 548 MB file pages in on demand. There is no `torch` and no `transformers`
dependency anywhere in this repository.

The tokenizer is also implemented from the published vocabulary and merge list, for the same reason:
the merge sequence is the thing worth showing, and a library call hides it.

---

## What is captured

Per example, from one instrumented forward pass:

| Captured | Shape / detail |
|---|---|
| Tokenization | Every pre-token, its bytes, its byte-encoded form, and **every merge** with the rank that selected it |
| Token / position / summed embeddings | First 24 of 768 dimensions, plus true vector norms |
| Nearest neighbours | Cosine similarity against all 50,257 embedding rows |
| Attention | **All 144 matrices** (12 layers × 12 heads), rounded to 4 dp |
| Head patterns | Auto-classified: previous token, current token, attention sink, broad, mixed |
| Worked example | One head's complete arithmetic — q·k, scaling, mask, exponentials, softmax, weighted sum |
| MLP | Top activated neurons of 3072, and what fraction are active |
| Logit lens | The decoded prediction after **every** layer |
| Output | Top-8 tokens with probabilities, and the distribution's entropy |

Payloads are 42–78 KB per example. Attention dominates: 12 × 12 × T² floats, which is why the
examples are kept to 5–8 tokens.

---

## The nine components

Ordered as the data flows: **tokenization → token embeddings → positional encoding → layer
normalisation → self-attention → multi-head attention → feed-forward → residual stream →
unembedding**.

Each carries a summary, a step-by-step account, the formula, *why it exists*, a commonly-held
misconception, and a pointer to what to look at in the data — that last field matters most, because a
heatmap nobody knows how to read teaches nothing.

---

## The ten examples

| id | text | chosen because |
|---|---|---|
| `cat-mat` | The cat sat on the mat | Baseline — the whole pipeline with nothing unusual |
| `counting` | 1, 2, 3, 4, | Confidence: 87% on ' 5', entropy collapses |
| `capital-france` | The capital of France is | Factual recall — and an honest failure, GPT-2 small ranks ' the' above ' Paris' |
| `subword` | unbelievable results | BPE: un + bel + iev + able, and the merge order that explains it |
| `coreference` | The doctor told the nurse that she | Pronoun resolution across candidates |
| `agreement` | The keys to the cabinet are | Long-range syntax — agrees with 'keys', not the nearer 'cabinet' |
| `code` | def add(a, b): return | Tokenization is tuned for prose; punctuation fragments |
| `repetition` | the cat the cat the cat the | **The clearest pattern in the set** — a head puts >99% on the token that followed the last occurrence |
| `rare-word` | The antidisestablishmentarianism debate | How far subword splitting goes |
| `analogy` | Paris is to France as Rome is to | Pattern matching and recall together |

---

## Correctness

The numpy implementation is checked against behaviour GPT-2 is known to have:

- `1, 2, 3, 4,` → **' 5' at 87.1%**
- `The cat sat on the` → ' floor', ' bed', ' couch'
- Every attention row sums to 1.0
- The masked upper triangle is exactly 0.0
- Softmax probabilities sum to 1.0, asserted at build time and shown in the UI

If the forward pass were wrong, none of these would hold.

---

## API reference

All paths are mounted under `/insidellm` and require the platform auth cookie.

| Method | Path | Returns |
|---|---|---|
| `GET` | `/overview` | Model facts and an honest description of the precompute method |
| `GET` | `/components` | The nine components, in data-flow order |
| `GET` | `/models` | GPT-2 / BERT / LLaMA specs and the comparison table |
| `GET` | `/examples` | The index — id, text, token count, top prediction |
| `GET` | `/examples/{id}` | One example's full capture |
| `GET` | `/positional` | GPT-2's learned position vectors, plus sinusoidal and rotary |

Example payloads are returned as **the bytes already on disk** — never parsed into Python and
re-serialised. On a two-core box that saved work is the difference between trivial and noticeable.

---

## Rebuilding the data

Only needed when the examples change.

```bash
# 1. Fetch the weights and tokenizer (once, ~550 MB, not committed)
mkdir -p /tmp/gpt2 && cd /tmp/gpt2
for f in model.safetensors vocab.json merges.txt config.json; do
  curl -L -O "https://huggingface.co/gpt2/resolve/main/$f"
done

# 2. From the repository root
pip install numpy regex
python -m projects.insidellm.precompute.build --weights /tmp/gpt2 --out projects/insidellm/data
```

Takes about half a minute. The output in `data/` **is committed** — it is what the service serves,
and rebuilding it must never be a deploy-time step.

---

## Design decisions

| Decision | Why |
|---|---|
| Precompute everything | The 2 GB host cannot hold a model. Fixed examples make it possible, and better to teach with |
| No typed input | Also the reason precomputation works. An arbitrary sentence usually demonstrates nothing |
| Implement GPT-2 by hand | A library returns an answer and throws away the intermediates, which are the whole subject |
| No torch anywhere | numpy plus a 20-line safetensors reader is enough, and keeps the repo installable in seconds |
| Serve raw bytes | Parsing JSON only to re-serialise it is pure waste on two cores |
| GPT-2 small, not a modern model | Its architecture is the one everything else varies from, and it is small enough to show *completely* |
| Ship `precompute/` with the project | The numpy forward pass is the clearest available statement of what a transformer does |
| Round attention to 4 dp | Far finer than a heatmap can render; the alternative triples the payload for nothing |
| No database, no cleanup hook | Read-only static content, so the project registers neither — the simplest thing the registry can host |

---

## What this project does not claim

- **Attention weights are not explanations.** They show where information was read from. That is a
  real constraint on what the model *could* have used, and it is not the same as why the answer came
  out as it did.
- **Head labels are descriptions, not roles.** Nothing assigned head 11 of layer 4 its job during
  training. The labels were inferred from the matrices afterwards, and the same head behaves
  differently on a different sentence.
- **GPT-2 small is not a current model.** It is orders of magnitude smaller than anything deployed
  today. It was chosen for architectural centrality and completeness of display, not capability.
