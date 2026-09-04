"""Turn the ten examples into the JSON the server ships.

Run once, by hand, whenever the examples change:

    python -m projects.insidellm.precompute.build --weights <dir> --out <dir>

Everything expensive happens here: loading 548 MB of weights, twelve forward
passes, nearest-neighbour searches over the full 50,257-token embedding table.
The output is a few hundred kilobytes per example, and that is all the running
service ever touches.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from projects.insidellm.examples import EXAMPLES
from projects.insidellm.precompute.gpt2_numpy import GPT2, softmax
from projects.insidellm.precompute.tokenizers import GPT2Tokenizer

# How much of each 768-dimension vector to ship. Enough to see structure and
# convince a reader it is real; shipping all of it would be 30x the size and no
# more informative on screen.
DIMS_SHOWN = 24

# Dimensions shown per vector in the worked arithmetic. The dot product is
# always computed over all 64 — only the display is truncated.
WORKED_DIMS = 6


def r(value, places=4):
    """Round for transport. JSON has no float32, and four decimal places is well
    past what any of these visualisations can render."""
    if isinstance(value, np.ndarray):
        return np.round(value.astype(np.float64), places).tolist()
    return round(float(value), places)


def attention_pattern(probs: np.ndarray) -> str:
    """Name what a head is doing, from its attention matrix alone.

    These are the patterns that recur across trained transformers and that a
    reader can verify by eye against the heatmap, so labelling them turns a wall
    of 144 grids into something navigable.
    """
    length = probs.shape[0]
    if length < 2:
        return "single-token"
    rows = probs[1:]  # position 0 can only attend to itself; it says nothing

    first_column = rows[:, 0].mean()
    diagonal = np.array([rows[i, i + 1] for i in range(len(rows))]).mean()
    previous = np.array([rows[i, i] for i in range(len(rows))]).mean()

    if first_column > 0.5:
        return "attention sink"
    if previous > 0.4:
        return "previous token"
    if diagonal > 0.4:
        return "current token"
    entropy = float(-(rows * np.log(rows + 1e-10)).sum(-1).mean())
    if entropy > np.log(length) * 0.8:
        return "broad / averaging"
    return "mixed"


def row_entropy(probs: np.ndarray) -> float:
    rows = probs[1:] if probs.shape[0] > 1 else probs
    return float(-(rows * np.log(rows + 1e-10)).sum(-1).mean())


def nearest_tokens(model: GPT2, tokenizer: GPT2Tokenizer, token_id: int, k: int = 6):
    """Nearest neighbours in embedding space — what the model considers similar
    to this token before any context is applied."""
    table = np.asarray(model.w["wte.weight"], dtype=np.float32)
    vector = table[token_id]
    norms = np.linalg.norm(table, axis=1)
    similarity = (table @ vector) / (norms * np.linalg.norm(vector) + 1e-8)
    similarity[token_id] = -np.inf
    top = np.argsort(-similarity)[:k]
    return [{"token": tokenizer.token_text(int(i)), "id": int(i), "similarity": r(similarity[i], 3)}
            for i in top]


def project_2d(vectors: np.ndarray):
    """Two principal components, so a set of 768-dimension vectors can be shown
    as a scatter plot without pretending the axes mean anything in particular."""
    centred = vectors - vectors.mean(axis=0, keepdims=True)
    if centred.shape[0] < 2:
        return [[0.0, 0.0]]
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    return r(centred @ components[:2].T, 3)


def top_predictions(logits: np.ndarray, tokenizer: GPT2Tokenizer, k: int = 8):
    probs = softmax(logits)
    top = np.argsort(-probs)[:k]
    return [{"token": tokenizer.token_text(int(i)), "id": int(i), "probability": r(probs[i], 5)}
            for i in top]


def build_worked_example(trace, tokenizer, tokens, model):
    """The full attention computation for one head, in numbers small enough to read.

    A reader cannot follow a 64-dimension dot product, but they can follow the
    shape of it: here are some of the numbers going in, here is the single value
    that comes out, here is what the scaling and the softmax do to it. The head
    is chosen rather than fixed — whichever one makes the most decisive choice at
    the final token is the one worth walking through.
    """
    query_position = len(tokens) - 1
    if query_position < 1:
        return None

    # Pick the most decisive head at the last position, ignoring attention to
    # position 0, which is usually a sink rather than a real decision.
    best = (0, 0, -1.0)
    for layer_index, layer in enumerate(trace["layers"]):
        probs = layer["attention"]["probs"][:, query_position, :]
        for head in range(probs.shape[0]):
            row = probs[head].copy()
            if len(row) > 1:
                row[0] = 0
            peak = float(row.max())
            if peak > best[2]:
                best = (layer_index, head, peak)
    layer_index, head, _ = best

    attention = trace["layers"][layer_index]["attention"]
    q = attention["q"][head, query_position]
    k_all = attention["k"][head]
    v_all = attention["v"][head]
    raw = attention["raw_scores"][head, query_position]
    scaled = attention["scaled_scores"][head, query_position]
    probs = attention["probs"][head, query_position]

    head_dim = q.shape[0]
    exponentials = np.exp(scaled[:query_position + 1] - scaled[:query_position + 1].max())

    return {
        "layer": layer_index,
        "head": head,
        "query_position": query_position,
        "query_token": tokens[query_position],
        "head_dim": int(head_dim),
        "dims_shown": WORKED_DIMS,
        "query_vector": r(q[:WORKED_DIMS], 3),
        "query_norm": r(np.linalg.norm(q), 3),
        "keys": [
            {
                "position": position,
                "token": tokens[position],
                "vector": r(k_all[position][:WORKED_DIMS], 3),
                "dot_product": r(raw[position], 3),
                "scaled": r(scaled[position], 3),
                "masked": position > query_position,
                # Six places, not four: the losing positions are genuinely of
                # order 1e-5, and rounding them to 0.0 next to a sum of 1.0
                # makes correct arithmetic look broken.
                "exponential": r(exponentials[position], 6) if position <= query_position else 0.0,
                "probability": r(probs[position], 6),
                "value_vector": r(v_all[position][:WORKED_DIMS], 3),
            }
            for position in range(len(tokens))
        ],
        "scale_divisor": r(np.sqrt(head_dim), 3),
        "exponential_sum": r(exponentials.sum(), 4),
        "output_vector": r((probs @ v_all)[:WORKED_DIMS], 3),
        "checks": {
            "probabilities_sum": r(probs.sum(), 6),
            "masked_positions_are_zero": bool(np.all(probs[query_position + 1:] == 0)) if query_position + 1 < len(tokens) else True,
        },
    }


def build_example(model: GPT2, tokenizer: GPT2Tokenizer, example: dict) -> dict:
    encoded = tokenizer.encode(example["text"])
    ids = encoded["ids"]
    trace = model.forward(ids)
    tokens = encoded["tokens"]

    token_embeddings = trace["token_embeddings"]
    position_embeddings = trace["position_embeddings"]
    embedding_sum = trace["embedding_sum"]

    # Attention: every layer, every head. This is the bulk of the payload, so it
    # is rounded hard — four places is far finer than a heatmap can show.
    attention_layers, patterns, entropies = [], [], []
    for layer in trace["layers"]:
        probs = layer["attention"]["probs"]
        attention_layers.append([r(probs[head], 4) for head in range(probs.shape[0])])
        patterns.append([attention_pattern(probs[head]) for head in range(probs.shape[0])])
        entropies.append([r(row_entropy(probs[head]), 3) for head in range(probs.shape[0])])

    # What the model would predict if it stopped at each layer.
    lens = []
    for layer_index, hidden in enumerate(trace["residual_stream"]):
        logits = model.logit_lens(hidden)
        lens.append({
            "layer": layer_index,
            "label": "embeddings" if layer_index == 0 else f"after layer {layer_index}",
            "top": top_predictions(logits[-1], tokenizer, 5),
        })

    mlp_layers = []
    for layer in trace["layers"]:
        activated = layer["mlp"]["activated"][-1]
        top = np.argsort(-np.abs(activated))[:8]
        mlp_layers.append({
            "layer": layer["index"],
            "expanded_dim": int(layer["mlp"]["expanded_dim"]),
            "top_neurons": [{"neuron": int(i), "activation": r(activated[i], 3)} for i in top],
            "fraction_active": r(float((activated > 0).mean()), 3),
            "mean_absolute": r(float(np.abs(activated).mean()), 3),
        })

    return {
        "id": example["id"],
        "text": example["text"],
        "title": example["title"],
        "teaches": example["teaches"],
        "look_for": example["look_for"],
        "model": {"name": "GPT-2 small", "layers": model.n_layer, "heads": model.n_head,
                  "d_model": model.n_embd, "head_dim": model.head_dim, "vocab": 50257},
        "tokens": [{"id": int(token_id), "text": text, "position": position}
                   for position, (token_id, text) in enumerate(zip(ids, tokens))],
        "tokenization": {
            "pieces": [
                {
                    "pre_token": piece["pre_token"],
                    "bytes": piece["bytes"],
                    "byte_encoded": piece["byte_encoded"],
                    "start_symbols": piece["start_symbols"],
                    "merges": piece["merges"],
                    "final_symbols": piece["final_symbols"],
                    "token_ids": piece["token_ids"],
                }
                for piece in encoded["pieces"]
            ],
            "total_tokens": len(ids),
            "total_characters": len(example["text"]),
        },
        "embeddings": {
            "dims_shown": DIMS_SHOWN,
            "d_model": model.n_embd,
            "token": r(token_embeddings[:, :DIMS_SHOWN], 3),
            "position": r(position_embeddings[:, :DIMS_SHOWN], 3),
            "sum": r(embedding_sum[:, :DIMS_SHOWN], 3),
            "norms": {
                "token": r(np.linalg.norm(token_embeddings, axis=1), 2),
                "position": r(np.linalg.norm(position_embeddings, axis=1), 2),
                "sum": r(np.linalg.norm(embedding_sum, axis=1), 2),
            },
            "neighbors": [nearest_tokens(model, tokenizer, int(token_id)) for token_id in ids],
            "projection": project_2d(token_embeddings),
        },
        "attention": {"layers": attention_layers, "patterns": patterns, "entropy": entropies},
        "worked_example": build_worked_example(trace, tokenizer, tokens, model),
        "mlp": {"layers": mlp_layers},
        "logit_lens": lens,
        "prediction": {
            "top": top_predictions(trace["logits"][-1], tokenizer, 8),
            "entropy": r(float(-(softmax(trace["logits"][-1]) *
                                 np.log(softmax(trace["logits"][-1]) + 1e-10)).sum()), 3),
        },
    }


def build_position_study(model: GPT2) -> dict:
    """How GPT-2's learned positional embeddings are actually organised, next to
    the sinusoidal scheme from the original paper and the rotary scheme used by
    most recent models."""
    wpe = np.asarray(model.w["wpe.weight"], dtype=np.float32)[:32]
    norms = np.linalg.norm(wpe, axis=1, keepdims=True)
    similarity = (wpe @ wpe.T) / (norms * norms.T + 1e-8)

    positions, dim = 32, 64
    index = np.arange(positions)[:, None]
    frequency = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))[None, :]
    sinusoidal = np.zeros((positions, dim), dtype=np.float32)
    sinusoidal[:, 0::2] = np.sin(index * frequency)
    sinusoidal[:, 1::2] = np.cos(index * frequency)

    rope_angles = index * frequency  # the angle each 2-D pair is rotated by

    return {
        "learned": {
            "source": "GPT-2's trained wpe matrix",
            "positions": positions,
            "dims_shown": DIMS_SHOWN,
            "vectors": r(wpe[:, :DIMS_SHOWN], 3),
            "similarity": r(similarity, 3),
            "norms": r(np.linalg.norm(wpe, axis=1), 2),
        },
        "sinusoidal": {
            "source": "Computed from the formula in 'Attention Is All You Need'",
            "positions": positions,
            "dims_shown": DIMS_SHOWN,
            "vectors": r(sinusoidal[:, :DIMS_SHOWN], 3),
        },
        "rotary": {
            "source": "Rotation angles per position for the first dimension pairs",
            "positions": positions,
            "angles": r(rope_angles[:, :8], 4),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Precompute the Inside an LLM artifacts")
    parser.add_argument("--weights", required=True, help="Directory with model.safetensors, vocab.json, merges.txt")
    parser.add_argument("--out", required=True, help="Directory to write JSON into")
    arguments = parser.parse_args()

    weights_dir = Path(arguments.weights)
    out_dir = Path(arguments.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2Tokenizer(weights_dir / "vocab.json", weights_dir / "merges.txt")
    model = GPT2(weights_dir / "model.safetensors")

    index = []
    for example in EXAMPLES:
        built = build_example(model, tokenizer, example)
        path = out_dir / f"{example['id']}.json"
        path.write_text(json.dumps(built, separators=(",", ":")), encoding="utf-8")
        size_kb = path.stat().st_size / 1024
        index.append({
            "id": example["id"], "text": example["text"], "title": example["title"],
            "teaches": example["teaches"], "tokens": len(built["tokens"]),
            "top_prediction": built["prediction"]["top"][0],
        })
        print(f"  {example['id']:<16} {len(built['tokens']):>2} tokens  {size_kb:>6.0f} KB")

    (out_dir / "index.json").write_text(json.dumps({"examples": index}, indent=1), encoding="utf-8")
    (out_dir / "positional.json").write_text(
        json.dumps(build_position_study(model), separators=(",", ":")), encoding="utf-8")

    total = sum(path.stat().st_size for path in out_dir.glob("*.json")) / 1024
    print(f"\nwrote {len(index)} examples + index + positional study — {total:.0f} KB total")


if __name__ == "__main__":
    main()
