"""The ten fixed examples.

Fixed rather than user-supplied for two reasons. The obvious one is cost: every
number this project displays is precomputed at build time, so the server never
loads a model and runs comfortably in a fraction of a gigabyte. The better one
is pedagogy — each sentence is chosen because it makes one specific mechanism
visible, and an arbitrary sentence usually makes none of them visible.
"""

EXAMPLES = [
    {
        "id": "cat-mat",
        "text": "The cat sat on the mat",
        "title": "A plain sentence",
        "teaches": "The whole pipeline end to end, with nothing unusual happening.",
        "look_for": "Every word is a single token, and the leading space is part of the token — "
                    "' cat' and 'cat' are different entries in the vocabulary.",
    },
    {
        "id": "counting",
        "text": "1, 2, 3, 4,",
        "title": "Continuing a pattern",
        "teaches": "Induction: attention finding a repeating structure and extending it.",
        "look_for": "The model is overwhelmingly confident about the next token. Later-layer heads "
                    "attend back to the earlier numbers rather than to the adjacent comma.",
    },
    {
        "id": "capital-france",
        "text": "The capital of France is",
        "title": "Recalling a fact",
        "teaches": "Where factual knowledge lives — the feed-forward layers, not attention.",
        "look_for": "Watch the logit lens: 'Paris' climbs through the middle layers, which is where "
                    "the MLP blocks contribute most.",
    },
    {
        "id": "subword",
        "text": "unbelievable results",
        "title": "One word, four tokens",
        "teaches": "Byte-pair encoding: how a word the vocabulary lacks gets built from pieces.",
        "look_for": "'unbelievable' becomes un + bel + iev + able. The merge order is by training "
                    "frequency, so 'able' forms long before 'iev'.",
    },
    {
        "id": "coreference",
        "text": "The doctor told the nurse that she",
        "title": "Working out who 'she' is",
        "teaches": "Attention resolving a pronoun against candidate antecedents.",
        "look_for": "At the final token, heads in the middle layers put weight on 'doctor' and "
                    "'nurse' rather than on the immediately preceding word.",
    },
    {
        "id": "agreement",
        "text": "The keys to the cabinet are",
        "title": "Agreement across a distance",
        "teaches": "Long-range syntax: the verb agrees with 'keys', not the nearer 'cabinet'.",
        "look_for": "Attention at 'are' reaches back over 'cabinet' to 'keys' — the grammatically "
                    "controlling word, not the closest one.",
    },
    {
        "id": "code",
        "text": "def add(a, b): return",
        "title": "Code instead of prose",
        "teaches": "That tokenization is tuned for text, and punctuation-heavy input fragments.",
        "look_for": "Symbols each become their own token, so a short line of code costs more "
                    "tokens than a longer English sentence.",
    },
    {
        "id": "repetition",
        "text": "the cat the cat the cat the",
        "title": "Deliberate repetition",
        "teaches": "Induction heads: 'this happened before, so it happens again'.",
        "look_for": "Sharp diagonal stripes in the attention maps, offset by the length of the "
                    "repeating unit. This is the clearest induction pattern in the set.",
    },
    {
        "id": "rare-word",
        "text": "The antidisestablishmentarianism debate",
        "title": "A very rare word",
        "teaches": "How far subword splitting goes when a word is genuinely unusual.",
        "look_for": "One English word costs many tokens. Token count is about familiarity to the "
                    "tokenizer, not length or difficulty.",
    },
    {
        "id": "analogy",
        "text": "Paris is to France as Rome is to",
        "title": "An analogy",
        "teaches": "Several mechanisms at once: pattern matching, factual recall, and copying.",
        "look_for": "The model must notice the A:B::C:? shape and retrieve a fact. Both the "
                    "attention maps and the logit lens are doing visible work here.",
    },
]

EXAMPLES_BY_ID = {example["id"]: example for example in EXAMPLES}
