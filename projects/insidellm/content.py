"""The written explanations.

Static text, served alongside the precomputed numbers. Each component says what
the operation is, why it exists at all, and what to look at in the data — the
last one matters most, because a heatmap nobody knows how to read teaches
nothing.
"""

COMPONENTS = [
    {
        "id": "tokenization",
        "order": 1,
        "name": "Tokenization",
        "tagline": "Text becomes a list of integers",
        "summary": "A model cannot read characters. Before anything else, text is cut into tokens "
                   "from a fixed vocabulary and each one is replaced by its index.",
        "how": [
            "The text is split on a regular expression, which keeps a leading space attached to the "
            "word that follows it. This is why ' cat' and 'cat' are two different tokens.",
            "Each fragment is converted to raw UTF-8 bytes, then those bytes are mapped to printable "
            "characters so the next step has something to work with.",
            "Byte-pair encoding repeatedly merges the most frequent adjacent pair. The merge list was "
            "learned from the training corpus and is applied strictly in rank order.",
            "Whatever symbols remain are looked up in the 50,257-entry vocabulary to give token IDs.",
        ],
        "why": "A fixed vocabulary of whole words would be enormous and would still fail on the first "
               "unfamiliar name. Working from bytes upward means any input can be represented — a "
               "rare word simply costs more tokens.",
        "formula": "text → pre-tokens → bytes → merge by rank → token IDs",
        "misconception": "Tokens are not words and not syllables. They are whatever the merge "
                         "statistics produced, which is why 'unbelievable' splits into un + bel + "
                         "iev + able rather than un + believe + able.",
        "look_at": "The merge trace. Every merge shows the pair, the rank that selected it, and the "
                   "state of the word afterwards. Low ranks are common pairs and always go first.",
    },
    {
        "id": "embeddings",
        "order": 2,
        "name": "Token embeddings",
        "tagline": "Each integer becomes a vector",
        "summary": "Every token ID indexes a row of a learned table, turning a meaningless integer "
                   "into 768 numbers that encode what the token tends to mean.",
        "how": [
            "The embedding matrix is 50,257 × 768 — one row per vocabulary entry.",
            "Lookup is exactly that: a table lookup. No arithmetic happens here.",
            "The values are learned during training, so tokens used in similar contexts end up "
            "with similar vectors.",
        ],
        "why": "Token ID 3797 is not 3797 times more anything than ID 1. The integers carry no "
               "meaning, so they are replaced by vectors positioned in a space where distance "
               "corresponds to similarity of use.",
        "formula": "embedding = W_e[token_id]     W_e ∈ ℝ^(50257 × 768)",
        "misconception": "This vector is not the token's meaning in context. It is the same every "
                         "time that token appears, whatever the sentence. Context arrives later, "
                         "in the attention layers.",
        "look_at": "The nearest neighbours. These are the tokens closest in embedding space before "
                   "any context is applied, so they show what the table alone knows.",
    },
    {
        "id": "positional",
        "order": 3,
        "name": "Positional encoding",
        "tagline": "Telling the model what came first",
        "summary": "Attention has no inherent sense of order, so position has to be added to the "
                   "input explicitly.",
        "how": [
            "GPT-2 learns a second table, one row per position, and simply adds it to the token "
            "embedding.",
            "The original Transformer paper instead computed fixed sine and cosine waves of "
            "different frequencies.",
            "Most recent models use rotary embeddings (RoPE), which rotate pairs of dimensions by an "
            "angle proportional to position, injecting position inside attention rather than before it.",
        ],
        "why": "Attention computes a weighted sum over all positions. Without positional information "
               "'the cat sat' and 'sat the cat' would produce identical results — the mechanism is "
               "order-blind by construction.",
        "formula": "x = W_e[token] + W_p[position]        (GPT-2, learned and added)",
        "misconception": "Adding position to meaning sounds like it should corrupt the meaning. In "
                         "768 dimensions there is room for both, and the following layers are "
                         "trained to read them apart.",
        "look_at": "The similarity matrix of GPT-2's learned position vectors. Nobody specified that "
                   "nearby positions should be similar — that structure emerged from training.",
    },
    {
        "id": "layernorm",
        "order": 4,
        "name": "Layer normalisation",
        "tagline": "Rescaling before every block",
        "summary": "Each vector is recentred to mean zero and unit variance, then rescaled by "
                   "learned parameters. It runs before attention and before the feed-forward block.",
        "how": [
            "Compute the mean and variance across the 768 dimensions of a single token's vector.",
            "Subtract the mean, divide by the standard deviation.",
            "Multiply by a learned gain and add a learned bias, so the layer can undo the "
            "normalisation where that is useful.",
        ],
        "why": "Residual connections add to the same running vector at every layer, so its magnitude "
               "would grow without bound. Normalising first keeps every block receiving inputs on a "
               "predictable scale, which is what makes deep stacks trainable.",
        "formula": "LN(x) = γ · (x − μ) / √(σ² + ε) + β",
        "misconception": "It normalises across the features of one token, not across the tokens in "
                         "the batch. Each token is normalised entirely on its own.",
        "look_at": "The mean and standard deviation recorded per token per layer — how far the "
                   "residual stream had drifted before it was pulled back.",
    },
    {
        "id": "attention",
        "order": 5,
        "name": "Self-attention",
        "tagline": "Every token looks at the tokens before it",
        "summary": "Each token forms a query, compares it against every other token's key, and takes "
                   "a weighted average of their values. This is the only place information moves "
                   "between positions.",
        "how": [
            "Project each token's vector three ways to get a query, a key and a value.",
            "Score every query against every key with a dot product — high where they align.",
            "Divide by √64. Without this the scores grow with dimension and softmax collapses to "
            "picking one position.",
            "Mask out future positions. A token predicting the next word must not see it.",
            "Softmax each row into weights that sum to 1, then take that weighted sum of the values.",
        ],
        "why": "Meaning depends on context, and context lives in other tokens. Attention is the "
               "mechanism that lets a token pull in exactly the other tokens it needs, with the "
               "choice of which ones learned rather than fixed.",
        "formula": "Attention(Q, K, V) = softmax( QKᵀ / √d_k + mask ) V",
        "misconception": "Attention weights are not an explanation of the model's reasoning. They "
                         "show where information was read from, which is a genuine constraint but "
                         "not the same as why the answer came out as it did.",
        "look_at": "The worked example. Every step of one head's computation at one position, with "
                   "the real numbers — dot product, scaling, masking, softmax, weighted sum.",
    },
    {
        "id": "multihead",
        "order": 6,
        "name": "Multi-head attention",
        "tagline": "Twelve of those, running side by side",
        "summary": "The 768 dimensions are split into 12 heads of 64, each doing its own attention. "
                   "Their outputs are concatenated and projected back.",
        "how": [
            "Split queries, keys and values into 12 independent 64-dimension slices.",
            "Run the full attention computation separately in each slice.",
            "Concatenate the 12 results back to 768 and apply one more learned projection.",
        ],
        "why": "One attention pattern per layer would force every relationship to compete for the "
               "same weights. Separate heads can specialise — and they demonstrably do, with some "
               "tracking the previous token and others matching repeated structure.",
        "formula": "MultiHead(x) = Concat(head₁ … head₁₂) W_O",
        "misconception": "Heads are not assigned roles. Nobody told head 11 of layer 4 to become an "
                         "induction head; the labels here were inferred from its behaviour after "
                         "the fact.",
        "look_at": "The grid of 144 heads. Each is labelled by the pattern it shows on this input — "
                   "and the same head can behave differently on a different sentence.",
    },
    {
        "id": "feedforward",
        "order": 7,
        "name": "Feed-forward network",
        "tagline": "Where the facts appear to live",
        "summary": "Each token is passed, independently, through a two-layer network that expands "
                   "768 dimensions to 3072 and back.",
        "how": [
            "Project up to 3072 dimensions.",
            "Apply GELU, a smooth activation that passes large values and suppresses negative ones.",
            "Project back down to 768.",
        ],
        "why": "Attention moves information between positions but is close to linear in what it does "
               "to it. The feed-forward block supplies the non-linear capacity, and it holds about "
               "two thirds of the model's parameters.",
        "formula": "FFN(x) = GELU(x W₁ + b₁) W₂ + b₂        768 → 3072 → 768",
        "misconception": "It looks like the boring part next to attention, but it is where most of "
                         "the parameters are, and interpretability work keeps locating specific "
                         "factual associations inside it.",
        "look_at": "Which of the 3072 neurons fire hardest at the final token, and how few of them "
                   "are active at once.",
    },
    {
        "id": "residual",
        "order": 8,
        "name": "Residual stream",
        "tagline": "Every block adds; nothing replaces",
        "summary": "Each block's output is added to its input rather than substituted for it, so a "
                   "single running vector carries information through all twelve layers.",
        "how": [
            "x = x + Attention(LayerNorm(x))",
            "x = x + FeedForward(LayerNorm(x))",
            "Repeat for all twelve blocks.",
        ],
        "why": "Addition gives gradients a direct path back to the start, which is what makes deep "
               "networks trainable at all. It also means each block can make a small correction "
               "instead of having to reconstruct everything it received.",
        "formula": "xₙ₊₁ = xₙ + Block(LayerNorm(xₙ))",
        "misconception": "The layers are not a pipeline that transforms and hands on. They all read "
                         "from and write to the same shared vector — which is precisely why the "
                         "logit lens works.",
        "look_at": "The logit lens. Because every block only adds, the running vector can be decoded "
                   "at any depth, and the prediction visibly sharpens layer by layer.",
    },
    {
        "id": "output",
        "order": 9,
        "name": "Unembedding and softmax",
        "tagline": "Back from vectors to a word",
        "summary": "The final vector is compared against every token in the vocabulary, producing one "
                   "score each, which softmax turns into probabilities.",
        "how": [
            "Apply a final layer normalisation.",
            "Multiply by the transpose of the embedding matrix — GPT-2 reuses the same table it used "
            "at the input.",
            "That gives 50,257 logits; softmax converts them into a distribution summing to 1.",
        ],
        "why": "Reusing the embedding matrix ties input and output representations together and "
               "saves 38 million parameters. A token's output score is literally the dot product of "
               "the final vector with that token's input embedding.",
        "formula": "logits = LayerNorm(x) · W_eᵀ        P = softmax(logits)",
        "misconception": "The model does not choose a word. It produces a distribution over all "
                         "50,257 tokens; picking one is a separate sampling decision made outside "
                         "the model.",
        "look_at": "The top predictions and the entropy. Low entropy means the model is confident; "
                   "on '1, 2, 3, 4,' it is very confident indeed.",
    },
]

MODELS = [
    {
        "id": "gpt2",
        "name": "GPT-2",
        "year": 2019,
        "family": "Decoder-only",
        "tagline": "The one running in this project",
        "parameters": "124M (small)",
        "spec": {"layers": 12, "heads": 12, "d_model": 768, "d_ff": 3072, "vocab": 50257,
                 "context": 1024, "head_dim": 64},
        "choices": {
            "Tokenizer": "Byte-level BPE, 50,257 tokens",
            "Position": "Learned embeddings, added to the input",
            "Attention": "Multi-head, causal mask, 12 full heads",
            "Normalisation": "LayerNorm, applied before each block",
            "Activation": "GELU",
            "Output": "Weights tied to the input embedding table",
        },
        "trained_for": "Predicting the next token on 40 GB of web text.",
        "why_it_matters": "It is the reference architecture. Nearly every decoder-only model since is "
                          "a variation on this skeleton, which is why it is the one worth learning "
                          "in full detail.",
    },
    {
        "id": "bert",
        "name": "BERT",
        "year": 2018,
        "family": "Encoder-only",
        "tagline": "Reads in both directions, predicts nothing next",
        "parameters": "110M (base)",
        "spec": {"layers": 12, "heads": 12, "d_model": 768, "d_ff": 3072, "vocab": 30522,
                 "context": 512, "head_dim": 64},
        "choices": {
            "Tokenizer": "WordPiece, 30,522 tokens, continuations marked '##'",
            "Position": "Learned embeddings, plus a segment embedding",
            "Attention": "Multi-head, no causal mask — every token sees every other",
            "Normalisation": "LayerNorm, applied after each block",
            "Activation": "GELU",
            "Output": "A classification head, not a vocabulary projection",
        },
        "trained_for": "Filling in masked words and judging whether two sentences follow one another.",
        "why_it_matters": "Removing the causal mask changes what the model is for. Seeing the whole "
                          "sentence at once makes it strong at classification and retrieval, and "
                          "unable to generate text left to right.",
    },
    {
        "id": "llama",
        "name": "LLaMA",
        "year": 2023,
        "family": "Decoder-only",
        "tagline": "The same skeleton, five components swapped",
        "parameters": "7B and up",
        "spec": {"layers": 32, "heads": 32, "d_model": 4096, "d_ff": 11008, "vocab": 32000,
                 "context": 4096, "head_dim": 128},
        "choices": {
            "Tokenizer": "SentencePiece BPE, 32,000 tokens",
            "Position": "Rotary (RoPE) — applied inside attention, not added at the input",
            "Attention": "Grouped-query: many query heads share fewer key/value heads",
            "Normalisation": "RMSNorm — no mean subtraction, no bias",
            "Activation": "SwiGLU, with a gate branch",
            "Output": "Untied output projection",
        },
        "trained_for": "Next-token prediction on trillions of tokens.",
        "why_it_matters": "It shows how little the core changed. Attention, residuals and the "
                          "feed-forward sandwich are identical to GPT-2; what moved is position "
                          "encoding, normalisation, the activation, and how keys and values are "
                          "shared to make inference cheaper.",
    },
]

COMPARISON_ROWS = [
    {"aspect": "Direction", "gpt2": "Left to right only", "bert": "Both directions",
     "llama": "Left to right only",
     "note": "The causal mask is the single line of code that separates a generator from an encoder."},
    {"aspect": "Position", "gpt2": "Learned, added at input", "bert": "Learned, added at input",
     "llama": "Rotary, applied in attention",
     "note": "RoPE encodes relative distance and extends past the trained context length far better."},
    {"aspect": "Normalisation", "gpt2": "LayerNorm before the block", "bert": "LayerNorm after the block",
     "llama": "RMSNorm before the block",
     "note": "Pre-norm made deep stacks trainable without a learning-rate warmup."},
    {"aspect": "Activation", "gpt2": "GELU", "bert": "GELU", "llama": "SwiGLU",
     "note": "SwiGLU adds a multiplicative gate, costing a third matrix for a consistent quality gain."},
    {"aspect": "Attention heads", "gpt2": "12 full heads", "bert": "12 full heads",
     "llama": "32 query heads, 8 key/value groups",
     "note": "Sharing key/value heads shrinks the memory the cache needs during generation."},
    {"aspect": "Output weights", "gpt2": "Tied to input embeddings", "bert": "Task-specific head",
     "llama": "Separate matrix",
     "note": "Tying saves parameters; at billions of parameters that saving stops mattering."},
]
