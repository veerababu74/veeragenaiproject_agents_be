"""GPT-2, implemented in numpy, with every intermediate value captured.

This is a real forward pass over the real pretrained weights — not a simulation.
It is written out longhand instead of calling a library because the whole point
is to expose the intermediates: a library returns an answer, and what this
project needs is the arithmetic that produced it.

Build-time only. The server never imports this, never loads the weights, and
never runs a model — it serves the JSON this produces.
"""

import json
import struct
from pathlib import Path

import numpy as np

DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "I64": np.int64}


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Minimal safetensors reader: an 8-byte header length, a JSON header, then
    the raw tensor bytes. Memory-mapped so the 548 MB file is paged in on demand
    rather than copied into RAM up front."""
    with open(path, "rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
    data_start = 8 + header_length
    blob = np.memmap(path, dtype=np.uint8, mode="r")

    tensors = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        start, end = spec["data_offsets"]
        raw = blob[data_start + start:data_start + end]
        tensors[name] = raw.view(DTYPES[spec["dtype"]]).reshape(spec["shape"])
    return tensors


def gelu(x: np.ndarray) -> np.ndarray:
    """GPT-2's 'gelu_new' — the tanh approximation, which is what the weights
    were trained against. The exact erf version gives slightly different numbers."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # Subtracting the max is what keeps exp() from overflowing; it does not
    # change the result because softmax is shift-invariant.
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    variance = x.var(axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(variance + eps)
    return normalized * weight + bias, mean.squeeze(-1), np.sqrt(variance + eps).squeeze(-1)


class GPT2:
    """The 124M-parameter model: 12 layers, 12 heads, 768 dimensions."""

    def __init__(self, weights_path: Path, n_layer=12, n_head=12, n_embd=768):
        self.w = load_safetensors(Path(weights_path))
        self.n_layer, self.n_head, self.n_embd = n_layer, n_head, n_embd
        self.head_dim = n_embd // n_head

    def block_weights(self, layer: int) -> dict:
        prefix = f"h.{layer}."
        return {key[len(prefix):]: value for key, value in self.w.items() if key.startswith(prefix)}

    def forward(self, ids: list[int]) -> dict:
        """Run the model and return every intermediate the visualiser needs."""
        tokens = np.asarray(ids, dtype=np.int64)
        positions = np.arange(len(ids))

        token_embeddings = np.asarray(self.w["wte.weight"][tokens], dtype=np.float32)
        position_embeddings = np.asarray(self.w["wpe.weight"][positions], dtype=np.float32)
        hidden = token_embeddings + position_embeddings

        trace = {
            "token_embeddings": token_embeddings,
            "position_embeddings": position_embeddings,
            "embedding_sum": hidden.copy(),
            "layers": [],
            "residual_stream": [hidden.copy()],
        }

        causal = np.triu(np.ones((len(ids), len(ids)), dtype=bool), k=1)

        for layer in range(self.n_layer):
            weights = self.block_weights(layer)
            captured: dict = {"index": layer}

            # ── attention ───────────────────────────────────────────────────
            normed, ln1_mean, ln1_std = layer_norm(hidden, weights["ln_1.weight"], weights["ln_1.bias"])
            captured["ln_1"] = {"output": normed.copy(), "mean": ln1_mean, "std": ln1_std}

            # GPT-2 stores these as Conv1D, so the weight is [in, out] and is
            # applied directly rather than transposed.
            qkv = normed @ weights["attn.c_attn.weight"] + weights["attn.c_attn.bias"]
            q, k, v = np.split(qkv, 3, axis=-1)

            def to_heads(x):
                return x.reshape(len(ids), self.n_head, self.head_dim).transpose(1, 0, 2)

            qh, kh, vh = to_heads(q), to_heads(k), to_heads(v)

            raw_scores = qh @ kh.transpose(0, 2, 1)
            # Scaling by sqrt(head_dim) keeps the dot products from growing with
            # dimension, which would push softmax into a near one-hot regime.
            scaled = raw_scores / np.sqrt(self.head_dim)
            masked = np.where(causal, -np.inf, scaled)
            probs = softmax(masked, axis=-1)
            head_out = probs @ vh

            merged = head_out.transpose(1, 0, 2).reshape(len(ids), self.n_embd)
            attn_out = merged @ weights["attn.c_proj.weight"] + weights["attn.c_proj.bias"]

            captured["attention"] = {
                "q": qh, "k": kh, "v": vh,
                "raw_scores": raw_scores, "scaled_scores": scaled, "probs": probs,
                "head_out": head_out, "merged": merged, "output": attn_out,
            }

            hidden = hidden + attn_out
            captured["after_attention_residual"] = hidden.copy()

            # ── feed-forward ────────────────────────────────────────────────
            normed2, ln2_mean, ln2_std = layer_norm(hidden, weights["ln_2.weight"], weights["ln_2.bias"])
            pre_activation = normed2 @ weights["mlp.c_fc.weight"] + weights["mlp.c_fc.bias"]
            activated = gelu(pre_activation)
            mlp_out = activated @ weights["mlp.c_proj.weight"] + weights["mlp.c_proj.bias"]

            captured["ln_2"] = {"output": normed2.copy(), "mean": ln2_mean, "std": ln2_std}
            captured["mlp"] = {
                "pre_activation": pre_activation, "activated": activated, "output": mlp_out,
                "expanded_dim": pre_activation.shape[-1],
            }

            hidden = hidden + mlp_out
            captured["after_mlp_residual"] = hidden.copy()

            trace["layers"].append(captured)
            trace["residual_stream"].append(hidden.copy())

        final, _, _ = layer_norm(hidden, self.w["ln_f.weight"], self.w["ln_f.bias"])
        # GPT-2 ties the output projection to the input embedding matrix: the
        # same table that turns ids into vectors turns vectors back into scores.
        logits = final @ np.asarray(self.w["wte.weight"], dtype=np.float32).T

        trace["final_norm"] = final
        trace["logits"] = logits
        return trace

    def logit_lens(self, hidden: np.ndarray, top_k: int = 5) -> np.ndarray:
        """Decode a mid-network residual stream as if it were the final layer.

        Because every block only *adds* to the residual stream, the stream is
        already in the space the unembedding reads — so the model's running
        guess can be read out at any depth, and watching it sharpen layer by
        layer is the clearest evidence that depth is doing something.
        """
        normed, _, _ = layer_norm(hidden, self.w["ln_f.weight"], self.w["ln_f.bias"])
        return normed @ np.asarray(self.w["wte.weight"], dtype=np.float32).T
