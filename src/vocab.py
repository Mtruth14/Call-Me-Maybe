"""Turn the entries of vocab.json into real text.

Qwen uses a byte-level BPE. Every byte is stored as a printable unicode
character (GPT-2 convention): "Ġ" is a space, "Ċ" is a newline, and a character
like "é" is stored as two such characters (its two UTF-8 bytes).
"""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path

_UNICODE_TO_BYTES_CACHE: dict[str, int] | None = None


def bytes_to_unicode() -> dict[int, str]:
    """Return the GPT-2 table: each byte value (0-255) -> one printable char."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def unicode_to_bytes() -> dict[str, int]:
    """Return the inverse of `bytes_to_unicode` (computed once, then cached)."""
    global _UNICODE_TO_BYTES_CACHE
    if _UNICODE_TO_BYTES_CACHE is None:
        b2u = bytes_to_unicode()
        _UNICODE_TO_BYTES_CACHE = {v: k for k, v in b2u.items()}
    return _UNICODE_TO_BYTES_CACHE


def decode_token(token: str) -> str | None:
    """Decode one vocabulary entry to text.

    Maps every char back to its byte, then decodes the bytes as UTF-8.

    Returns:
        The text, or None if the entry is not decodable on its own (a fragment
        of a multi-byte character, or an unknown char). Such tokens are never
        allowed during generation.
    """
    u2b = unicode_to_bytes()
    byte_sequence = bytearray()

    for char in token:
        if char not in u2b:
            return None
        byte_sequence.append(u2b[char])

    try:
        return byte_sequence.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None


class Vocabulary:
    """Decoded vocabulary: token id -> text, for the usable tokens only."""

    def __init__(self, texts: dict[int, str], size: int) -> None:
        """Args:
            texts: id -> decoded text, only for usable tokens.
            size: length of the logits vector returned by the model. It can be
                larger than vocab.json (special tokens): those ids have no text.
        """
        self._texts = texts
        self._size = size

    @classmethod
    def from_file(cls, path: str | Path, size: int) -> Vocabulary:
        """Load vocab.json and decode every entry with `decode_token`."""
        vocab_path = Path(path)
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts: dict[int, str] = {}
        if isinstance(data, dict):
            for token_str, token_id in data.items():
                if isinstance(token_id, int):
                    decoded = decode_token(str(token_str))
                    if decoded is not None:
                        texts[token_id] = decoded

        return cls(texts=texts, size=size)

    @property
    def size(self) -> int:
        """Length of the logits vector."""
        return self._size

    def text(self, token_id: int) -> str:
        """Text of a token, or "" if the token is not usable."""
        return self._texts.get(token_id, "")

    def items(self) -> Iterator[tuple[int, str]]:
        """Iterate over (id, text) of the usable tokens."""
        yield from self._texts.items()
