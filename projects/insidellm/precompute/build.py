"""Turn the ten examples into the JSON the server ships.

Run once, by hand, whenever the examples change:

    python projects/insidellm/precompute/build.py --weights <dir> --out <dir>

Everything expensive happens here: loading 548 MB of weights, twelve forward
passes, nearest-neighbour searches over the full 50,257-token embedding table.
The output is a few hundred kilobytes per example, and that is all the running
service ever touches.
"""

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np

# Run as a script, `projects.insidellm` would be imported through
# `projects/__init__.py`, which registers every project in the repository and so
# pulls in fastapi, jwt and langchain — none of which this build needs and all of
# which would make the documented "pip install numpy regex" a lie. Registering
# the two packages as bare namespaces first skips those __init__ files and
# imports only the four modules below.
if __name__ == "__main__" and "projects" not in sys.modules:
    ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(ROOT))
    for _name, _path in (("projects", ROOT / "projects"),
                         ("projects.insidellm", ROOT / "projects" / "insidellm")):
        _module = types.ModuleType(_name)
        _module.__path__ = [str(_path)]
        sys.modules[_name] = _module

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
    # The dot product and both norms travel with the similarity so the page can
    # show the division that produced it rather than asserting the quotient.
    return [{"token": tokenizer.token_text(int(i)), "id": int(i),
             "similarity": r(similarity[i], 3),
             "dot": r(float(table[i] @ vector), 2),
             "norm": r(float(norms[i]), 2),
             "query_norm": r(float(np.linalg.norm(vector)), 2)}
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
    # exp(logit - max) is the numerator the softmax actually divides; shipping it
    # alongside the logit lets the page show the arithmetic instead of the result.
    shifted = np.exp(logits - logits.max())
    return [{"token": tokenizer.token_text(int(i)), "id": int(i), "probability": r(probs[i], 5),
             "logit": r(logits[i], 3), "exp_shifted": r(shifted[i], 6)}
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
                # A 64-term dot product cannot be printed, but its first terms
                # can: these are q_i x k_i for the dimensions shown above, so a
                # reader can add them up and see the total is of that kind.
                "products": r(q[:WORKED_DIMS] * k_all[position][:WORKED_DIMS], 3),
                "partial_sum": r(float(q[:WORKED_DIMS] @ k_all[position][:WORKED_DIMS]), 3),
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


def build_layernorm(model: GPT2, trace, tokens) -> dict:
    """The normalisation arithmetic, at one token, with every term of the formula.

    LN(x) = gamma . (x - mu) / sqrt(sigma^2 + eps) + beta is four operations, and a
    reader can follow all four if the intermediates are present. The worked case
    is the first LayerNorm of layer 0 at the final token; the drift table is the
    same statistics at every layer, which is the evidence that the residual
    stream really does grow the way the explanation claims.
    """
    position = len(tokens) - 1
    x = trace["embedding_sum"][position]
    mean = float(x.mean())
    variance = float(x.var())
    eps = 1e-5
    std = float(np.sqrt(variance + eps))
    centred = x - mean
    normalised = centred / std
    weights = model.block_weights(0)
    gamma = np.asarray(weights["ln_1.weight"], dtype=np.float32)
    beta = np.asarray(weights["ln_1.bias"], dtype=np.float32)
    output = normalised * gamma + beta

    drift = []
    for entry in trace["layers"]:
        residual = trace["residual_stream"][entry["index"]][position]
        drift.append({
            "layer": entry["index"],
            "ln1_mean": r(float(entry["ln_1"]["mean"][position]), 3),
            "ln1_std": r(float(entry["ln_1"]["std"][position]), 3),
            "ln2_mean": r(float(entry["ln_2"]["mean"][position]), 3),
            "ln2_std": r(float(entry["ln_2"]["std"][position]), 3),
            "residual_norm": r(float(np.linalg.norm(residual)), 2),
        })

    return {
        "eps": eps,
        "dims_shown": WORKED_DIMS,
        "d_model": int(x.shape[0]),
        "position": position,
        "token": tokens[position],
        "site": "layer 0, before attention",
        "input": r(x[:WORKED_DIMS], 3),
        "mean": r(mean, 4),
        "variance": r(variance, 4),
        "std": r(std, 4),
        "centred": r(centred[:WORKED_DIMS], 3),
        "normalised": r(normalised[:WORKED_DIMS], 3),
        "gamma": r(gamma[:WORKED_DIMS], 3),
        "beta": r(beta[:WORKED_DIMS], 3),
        "output": r(output[:WORKED_DIMS], 3),
        "input_norm": r(float(np.linalg.norm(x)), 2),
        "output_norm": r(float(np.linalg.norm(output)), 2),
        # Proof the formula was applied and not merely quoted.
        "checks": {
            "normalised_mean": r(float(normalised.mean()), 6),
            "normalised_std": r(float(normalised.std()), 4),
        },
        "drift": drift,
    }


def build_residual(trace, tokens) -> dict:
    """What each block actually added to the running vector.

    The residual claim - every block adds a correction rather than replacing the
    stream - is checkable arithmetic: compare the norm of what a block wrote
    against the norm of the stream it wrote into. Doing that per layer is the
    difference between being told addition matters and seeing that attention
    contributes a few percent while the stream itself grows steadily.
    """
    position = len(tokens) - 1
    layers = []
    for entry in trace["layers"]:
        index = entry["index"]
        before = trace["residual_stream"][index][position]
        attention_out = entry["attention"]["output"][position]
        after_attention = entry["after_attention_residual"][position]
        mlp_out = entry["mlp"]["output"][position]
        after_mlp = entry["after_mlp_residual"][position]
        before_norm = float(np.linalg.norm(before))
        after_norm = float(np.linalg.norm(after_mlp))
        cosine = float(before @ after_mlp / (before_norm * after_norm + 1e-8))
        layers.append({
            "layer": index,
            "in_norm": r(before_norm, 2),
            "attention_norm": r(float(np.linalg.norm(attention_out)), 2),
            "after_attention_norm": r(float(np.linalg.norm(after_attention)), 2),
            "mlp_norm": r(float(np.linalg.norm(mlp_out)), 2),
            "out_norm": r(after_norm, 2),
            # What share of the outgoing stream each sub-block wrote.
            "attention_share": r(float(np.linalg.norm(attention_out)) / (after_norm + 1e-8), 3),
            "mlp_share": r(float(np.linalg.norm(mlp_out)) / (after_norm + 1e-8), 3),
            "direction_cosine": r(cosine, 3),
        })

    first = trace["layers"][0]
    return {
        "dims_shown": WORKED_DIMS,
        "position": position,
        "token": tokens[position],
        "layers": layers,
        "worked": {
            "layer": 0,
            "x": r(trace["residual_stream"][0][position][:WORKED_DIMS], 3),
            "attention_out": r(first["attention"]["output"][position][:WORKED_DIMS], 3),
            "sum": r(first["after_attention_residual"][position][:WORKED_DIMS], 3),
        },
    }


def build_prediction(model: GPT2, tokenizer: GPT2Tokenizer, trace) -> dict:
    """The last two operations, with the quantities they are computed from.

    A logit is a dot product between the final vector and one row of the
    embedding table, and softmax divides by a sum over all 50,257 of them.
    Neither is visible from a probability alone, so both the logits and the
    partition function they were divided by are shipped.
    """
    logits = trace["logits"][-1]
    final = trace["final_norm"][-1]
    table = np.asarray(model.w["wte.weight"], dtype=np.float32)
    probabilities = softmax(logits)
    maximum = float(logits.max())
    partition = float(np.exp(logits - maximum).sum())

    top = top_predictions(logits, tokenizer, 8)
    winner = int(top[0]["id"])
    embedding = table[winner]
    final_norm = float(np.linalg.norm(final))
    embedding_norm = float(np.linalg.norm(embedding))

    return {
        "top": top,
        "entropy": r(float(-(probabilities * np.log(probabilities + 1e-10)).sum()), 3),
        "vocab": int(logits.shape[0]),
        "max_logit": r(maximum, 3),
        # Every exponential in the payload was shifted by max_logit, so this is
        # the denominator that turns them into the probabilities shown.
        "partition": r(partition, 4),
        # A logit is |x| |e| cos(theta): the same dot product as everywhere else,
        # against the very embedding row the input lookup used.
        "unembedding": {
            "token": top[0]["token"],
            "id": winner,
            "final_norm": r(final_norm, 2),
            "embedding_norm": r(embedding_norm, 2),
            "cosine": r(float(final @ embedding / (final_norm * embedding_norm + 1e-8)), 4),
            "logit": r(float(final @ embedding), 3),
        },
    }


# How many dimensions of each vector the run view shows. Fewer than the
# component panels use: the run view puts two vectors side by side on one row
# and repeats that for ~80 rows, so the strip has to stay narrow enough to read.
TRACE_DIMS = 12


def build_trace(model: GPT2, tokenizer: GPT2Tokenizer, trace, tokens, ids) -> dict:
    """One token's journey through all twelve layers, stage by stage.

    The component panels answer "what does attention do?" and each one shows a
    different slice of a different thing. None of them answers "what happened to
    *this* vector, and then what happened next?" -- and that question is the one
    a reader actually starts with.

    So this records the same position at every stage boundary: what entered, what
    each sub-block wrote, and what the running sum became. The output of every
    stage here is literally the input of the next, which is what makes the chain
    followable rather than a set of unrelated readings.

    Only the final position is traced. It is the one the prediction is made from,
    and tracing all of them would multiply the payload by the token count for no
    extra insight.
    """
    position = len(tokens) - 1

    def strip(vector):
        return r(vector[:TRACE_DIMS], 3)

    def norm(vector):
        return r(float(np.linalg.norm(vector)), 2)

    def decode(hidden):
        """What the model would predict if it stopped here. Cheap, and it turns
        the residual stream from an opaque vector into something with a meaning
        the reader can watch change."""
        logits = model.logit_lens(hidden)
        probabilities = softmax(logits[-1])
        best = int(np.argmax(probabilities))
        return {"token": tokenizer.token_text(best), "probability": r(probabilities[best], 4)}

    layers = []
    for entry in trace["layers"]:
        index = entry["index"]
        incoming = trace["residual_stream"][index]
        attention = entry["attention"]

        # Which head moved the most information into this position, and from
        # where. Position 0 is excluded because it acts as a sink in most heads
        # and would win almost every layer without meaning anything.
        probabilities = attention["probs"][:, position, :]
        candidates = probabilities.copy()
        if candidates.shape[1] > 1:
            candidates[:, 0] = 0
        head = int(np.argmax(candidates.max(axis=1)))
        source = int(np.argmax(candidates[head]))

        activated = entry["mlp"]["activated"][position]
        pre_activation = entry["mlp"]["pre_activation"][position]
        neuron = int(np.argmax(np.abs(activated)))

        layers.append({
            "layer": index,
            "in": strip(incoming[position]),
            "in_norm": norm(incoming[position]),
            "ln1": {
                "mean": r(float(entry["ln_1"]["mean"][position]), 3),
                "std": r(float(entry["ln_1"]["std"][position]), 3),
                "output": strip(entry["ln_1"]["output"][position]),
                "output_norm": norm(entry["ln_1"]["output"][position]),
            },
            "attention": {
                "output": strip(attention["output"][position]),
                "output_norm": norm(attention["output"][position]),
                "head": head,
                "source_position": source,
                "source_token": tokens[source],
                "weight": r(float(probabilities[head, source]), 4),
                # How sharply that head was focused, in this layer, here.
                "entropy": r(float(-(probabilities[head] *
                                     np.log(probabilities[head] + 1e-10)).sum()), 3),
            },
            "after_attention": strip(entry["after_attention_residual"][position]),
            "after_attention_norm": norm(entry["after_attention_residual"][position]),
            "ln2": {
                "mean": r(float(entry["ln_2"]["mean"][position]), 3),
                "std": r(float(entry["ln_2"]["std"][position]), 3),
                "output": strip(entry["ln_2"]["output"][position]),
                "output_norm": norm(entry["ln_2"]["output"][position]),
            },
            "mlp": {
                "output": strip(entry["mlp"]["output"][position]),
                "output_norm": norm(entry["mlp"]["output"][position]),
                "neuron": neuron,
                "pre_activation": r(float(pre_activation[neuron]), 3),
                "activation": r(float(activated[neuron]), 3),
                "expanded_dim": int(entry["mlp"]["expanded_dim"]),
                "fraction_active": r(float((activated > 0).mean()), 3),
            },
            "out": strip(entry["after_mlp_residual"][position]),
            "out_norm": norm(entry["after_mlp_residual"][position]),
            # The running guess, after this layer has had its say.
            "prediction": decode(trace["residual_stream"][index + 1]),
        })

    final = trace["final_norm"][position]
    logits = trace["logits"][position]
    return {
        "position": position,
        "token": tokens[position],
        "token_id": int(ids[position]),
        "dims_shown": TRACE_DIMS,
        "d_model": int(final.shape[0]),
        "embedding": {
            "token": strip(trace["token_embeddings"][position]),
            "token_norm": norm(trace["token_embeddings"][position]),
            "position": strip(trace["position_embeddings"][position]),
            "position_norm": norm(trace["position_embeddings"][position]),
            "sum": strip(trace["embedding_sum"][position]),
            "sum_norm": norm(trace["embedding_sum"][position]),
            "prediction": decode(trace["residual_stream"][0]),
        },
        "layers": layers,
        "final_norm": {
            "output": strip(final),
            "output_norm": norm(final),
            "mean": r(float(trace["residual_stream"][-1][position].mean()), 3),
            "std": r(float(trace["residual_stream"][-1][position].std()), 3),
        },
        "output": {
            "max_logit": r(float(logits.max()), 3),
            "partition": r(float(np.exp(logits - logits.max()).sum()), 4),
            "top": top_predictions(logits, tokenizer, 5),
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
        pre_activation = layer["mlp"]["pre_activation"][-1]
        top = np.argsort(-np.abs(activated))[:8]
        mlp_layers.append({
            "layer": layer["index"],
            "expanded_dim": int(layer["mlp"]["expanded_dim"]),
            # Both sides of GELU. With only the output there is no way to show
            # what the activation did; with both, the curve becomes checkable.
            "top_neurons": [{"neuron": int(i), "activation": r(activated[i], 3),
                             "pre_activation": r(pre_activation[i], 3)} for i in top],
            "fraction_active": r(float((activated > 0).mean()), 3),
            "mean_absolute": r(float(np.abs(activated).mean()), 3),
        })

    prediction = build_prediction(model, tokenizer, trace)

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
        "layernorm": build_layernorm(model, trace, tokens),
        "attention": {"layers": attention_layers, "patterns": patterns, "entropy": entropies},
        "worked_example": build_worked_example(trace, tokenizer, tokens, model),
        "residual": build_residual(trace, tokens),
        "trace": build_trace(model, tokenizer, trace, tokens, ids),
        "mlp": {"layers": mlp_layers},
        "logit_lens": lens,
        "prediction": prediction,
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


def verify(built: dict) -> list[str]:
    """Re-derive every piece of arithmetic the walkthrough prints.

    The page now shows its working — (x - mu) / sigma with the actual numbers,
    exp(l - max) / Z with the actual logits — and a substitution that does not
    come out is worse than no substitution at all, because a reader who checks it
    by hand and finds it wrong has been taught something false.

    So every identity displayed on the page is recomputed here from the payload
    that will ship, and a mismatch fails the build. Tolerances are loose enough
    for four-decimal rounding and no looser.
    """
    problems = []

    def close(left, right, tolerance, message):
        if abs(left - right) > tolerance:
            problems.append(f"{message}: {left} vs {right}")

    normalisation = built["layernorm"]
    close(normalisation["normalised"][0],
          (normalisation["input"][0] - normalisation["mean"]) / normalisation["std"],
          0.01, "layer norm: x-hat does not follow from x, mu and sigma")
    close(normalisation["output"][0],
          normalisation["gamma"][0] * normalisation["normalised"][0] + normalisation["beta"][0],
          0.01, "layer norm: gamma x-hat + beta does not give the output")
    close(normalisation["checks"]["normalised_std"], 1.0, 0.01,
          "layer norm: normalised vector is not unit variance")

    residual = built["residual"]
    for index, (left, right, total) in enumerate(zip(
            residual["worked"]["x"], residual["worked"]["attention_out"], residual["worked"]["sum"])):
        close(total, left + right, 0.01, f"residual: the addition is wrong at dimension {index}")
    if residual["layers"][-1]["out_norm"] <= residual["layers"][0]["in_norm"]:
        problems.append("residual: the stream did not grow, so the LayerNorm argument breaks")

    prediction = built["prediction"]
    top = prediction["top"][0]
    close(top["exp_shifted"] / prediction["partition"], top["probability"], 0.001,
          "output: exp(l - max) / Z does not reproduce the probability")
    unembedding = prediction["unembedding"]
    close(unembedding["final_norm"] * unembedding["embedding_norm"] * unembedding["cosine"],
          unembedding["logit"], 0.6,
          "output: |z| |e| cos(theta) does not reproduce the logit")

    for layer in built["mlp"]["layers"]:
        for neuron in layer["top_neurons"]:
            h = neuron["pre_activation"]
            close(neuron["activation"],
                  0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h ** 3))),
                  0.01, f"feed-forward: GELU mismatch at neuron {neuron['neuron']}")

    for neighbours in built["embeddings"]["neighbors"]:
        for neighbour in neighbours:
            close(neighbour["dot"] / (neighbour["norm"] * neighbour["query_norm"]),
                  neighbour["similarity"], 0.01,
                  f"embeddings: the cosine for {neighbour['token']!r} does not divide out")

    # The run view's two loudest claims: that the stages form an unbroken chain,
    # and that the residual steps are additions the reader could do by hand.
    # Both are stated on screen, so neither is safe to leave unchecked.
    trace_data = built["trace"]
    previous = trace_data["embedding"]["sum"]
    for index_of, (token_part, position_part, total) in enumerate(zip(
            trace_data["embedding"]["token"], trace_data["embedding"]["position"], previous)):
        close(total, token_part + position_part, 0.01,
              f"trace: x0 is not token + position at dimension {index_of}")
    for entry in trace_data["layers"]:
        layer_index = entry["layer"]
        for index_of, (arriving, leaving) in enumerate(zip(entry["in"], previous)):
            close(arriving, leaving, 0.006,
                  f"trace: layer {layer_index} does not receive what layer "
                  f"{layer_index - 1} produced, at dimension {index_of}")
        for index_of, (stream, written, total) in enumerate(zip(
                entry["in"], entry["attention"]["output"], entry["after_attention"])):
            close(total, stream + written, 0.01,
                  f"trace: layer {layer_index} attention residual at dimension {index_of}")
        for index_of, (stream, written, total) in enumerate(zip(
                entry["after_attention"], entry["mlp"]["output"], entry["out"])):
            close(total, stream + written, 0.01,
                  f"trace: layer {layer_index} feed-forward residual at dimension {index_of}")
        previous = entry["out"]
    if trace_data["output"]["top"][0]["token"] != built["prediction"]["top"][0]["token"]:
        problems.append("trace: the run ends on a different token than the walkthrough reports")

    worked = built["worked_example"]
    if worked:
        visible = [key for key in worked["keys"] if not key["masked"]]
        winner = max(visible, key=lambda key: key["probability"])
        close(sum(winner["products"]), winner["partial_sum"], 0.02,
              "attention: the shown products do not sum to the partial")
        close(winner["dot_product"] / worked["scale_divisor"], winner["scaled"], 0.01,
              "attention: the scaling step is wrong")
        close(winner["exponential"] / worked["exponential_sum"], winner["probability"], 0.001,
              "attention: softmax does not reproduce the weight")

    return problems


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
        problems = verify(built)
        if problems:
            raise SystemExit(
                f"{example['id']}: the payload contradicts the arithmetic the page shows\n  "
                + "\n  ".join(problems))
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
