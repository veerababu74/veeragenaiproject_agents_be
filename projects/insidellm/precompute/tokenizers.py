"""Real tokenizers, implemented from their published vocabularies.

Written out rather than imported from a library because the point of this
project is to show what tokenization *does*, and a library call hides exactly
the steps worth showing. Each tokenizer here records its intermediate work so
the frontend can replay it: the byte encoding, every merge in the order it was
applied, and the rank that justified each one.

Build-time only. Nothing here runs on the server.
"""

import json
from pathlib import Path

import regex

# GPT-2's pre-tokenizer. It splits before byte-pair encoding so that merges can
# never cross a word boundary, and it deliberately keeps the leading space with
# the word that follows — which is why " cat" and "cat" are different tokens.
GPT2_PATTERN = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def byte_to_unicode() -> dict[int, str]:
    """GPT-2's byte<->unicode table.

    BPE operates on printable characters, but the input is raw bytes, so all 256
    byte values are mapped to visible codepoints first. The 188 already-printable
    ones map to themselves; the rest are shifted into an unused block. This is
    why a space shows up as 'Ġ' in the vocabulary.
    """
    printable = (list(range(ord("!"), ord("~") + 1))
                 + list(range(ord("\xa1"), ord("\xac") + 1))
                 + list(range(ord("\xae"), ord("\xff") + 1)))
    mapping = {byte: chr(byte) for byte in printable}
    spare = 0
    for byte in range(256):
        if byte not in mapping:
            mapping[byte] = chr(256 + spare)
            spare += 1
    return mapping


class GPT2Tokenizer:
    def __init__(self, vocab_path: Path, merges_path: Path):
        self.vocab: dict[str, int] = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        self.decoder = {index: token for token, index in self.vocab.items()}
        lines = Path(merges_path).read_text(encoding="utf-8").split("\n")[1:]
        self.ranks = {tuple(line.split()): rank for rank, line in enumerate(lines) if len(line.split()) == 2}
        self.byte_encoder = byte_to_unicode()
        self.byte_decoder = {char: byte for byte, char in self.byte_encoder.items()}

    def _merge_word(self, word: str) -> tuple[list[str], list[dict]]:
        """Byte-pair-encode one pre-token, recording each merge as it happens."""
        symbols = list(word)
        trace: list[dict] = []
        while len(symbols) > 1:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            ranked = [(self.ranks[pair], pair) for pair in pairs if pair in self.ranks]
            if not ranked:
                break
            # Lowest rank wins: merges are ordered by how often the pair occurred
            # in the training corpus, so the most common pair is always taken first.
            rank, best = min(ranked, key=lambda item: item[0])
            merged: list[str] = []
            index = 0
            while index < len(symbols):
                if (index < len(symbols) - 1
                        and symbols[index] == best[0] and symbols[index + 1] == best[1]):
                    merged.append(best[0] + best[1])
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1
            symbols = merged
            trace.append({"pair": [best[0], best[1]], "merged": best[0] + best[1],
                          "rank": rank, "result": list(symbols)})
        return symbols, trace

    def encode(self, text: str) -> dict:
        """Tokenize, returning both the result and the work that produced it."""
        pieces = []
        for pre_token in GPT2_PATTERN.findall(text):
            encoded = "".join(self.byte_encoder[byte] for byte in pre_token.encode("utf-8"))
            symbols, trace = self._merge_word(encoded)
            pieces.append({
                "pre_token": pre_token,
                "bytes": list(pre_token.encode("utf-8")),
                "byte_encoded": encoded,
                "start_symbols": list(encoded),
                "merges": trace,
                "final_symbols": symbols,
                "token_ids": [self.vocab[symbol] for symbol in symbols],
            })

        ids, tokens = [], []
        for piece in pieces:
            for symbol, token_id in zip(piece["final_symbols"], piece["token_ids"]):
                ids.append(token_id)
                tokens.append(self.decode_token(symbol))
        return {"pieces": pieces, "ids": ids, "tokens": tokens}

    def decode_token(self, symbol: str) -> str:
        return bytearray(self.byte_decoder[char] for char in symbol).decode("utf-8", errors="replace")

    def token_text(self, token_id: int) -> str:
        return self.decode_token(self.decoder[token_id])


class WordPieceTokenizer:
    """BERT's tokenizer: lowercase, split on punctuation, then greedy
    longest-match against the vocabulary, marking continuations with '##'."""

    def __init__(self, vocab_path: Path, lower: bool = True):
        self.vocab = {line: index for index, line in
                      enumerate(Path(vocab_path).read_text(encoding="utf-8").splitlines())}
        self.lower = lower

    def _split(self, text: str) -> list[str]:
        if self.lower:
            text = text.lower()
        return regex.findall(r"\w+|[^\w\s]", text)

    def encode(self, text: str) -> dict:
        pieces = []
        for word in self._split(text):
            # Greedy longest-match-first: take the longest prefix in the vocab,
            # then keep matching the remainder as '##' continuations.
            start, subwords, attempts = 0, [], []
            unknown = False
            while start < len(word):
                end = len(word)
                found = None
                while end > start:
                    candidate = word[start:end]
                    if start > 0:
                        candidate = "##" + candidate
                    attempts.append(candidate)
                    if candidate in self.vocab:
                        found = candidate
                        break
                    end -= 1
                if found is None:
                    unknown = True
                    subwords = ["[UNK]"]
                    break
                subwords.append(found)
                start = end
            pieces.append({"word": word, "subwords": subwords, "unknown": unknown,
                           "token_ids": [self.vocab.get(s, self.vocab.get("[UNK]", 100)) for s in subwords]})

        tokens = ["[CLS]"] + [s for piece in pieces for s in piece["subwords"]] + ["[SEP]"]
        ids = ([self.vocab["[CLS]"]]
               + [i for piece in pieces for i in piece["token_ids"]]
               + [self.vocab["[SEP]"]])
        return {"pieces": pieces, "tokens": tokens, "ids": ids}
