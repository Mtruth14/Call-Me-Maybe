"""Character-level automata that describe the text of ONE value.

An automaton knows which characters may come next. `MaskCache` turns that into a
boolean mask over the vocabulary: a token is allowed if the automaton accepts
all its characters in order.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
import json

import numpy as np
from numpy.typing import NDArray

from src.models import ParameterSpec
from src.vocab import Vocabulary

State = Hashable


class Automaton(ABC):
    """Abstract automaton for one value."""

    @property
    @abstractmethod
    def key(self) -> Hashable:
        """Identifies this automaton in the mask cache, e.g. ("number", False)."""

    @abstractmethod
    def start(self) -> State:
        """Initial state."""

    @abstractmethod
    def step(self, state: State, char: str) -> State | None:
        """Next state after `char`, or None if `char` is not allowed."""

    @abstractmethod
    def is_done(self, state: State) -> bool:
        """True if the value is complete and nothing more may be added."""

    def can_stop(self, state: State) -> bool:
        """True if the text so far is a valid value (default: same as is_done)."""
        return self.is_done(state)

    def forced_text(self, state: State) -> str | None:
        """Text that must come next if only one continuation is possible."""
        return None

    @abstractmethod
    def shortest_completion(self, state: State) -> str:
        """Shortest text that turns the current state into a valid value."""

    def consume(self, state: State, text: str) -> tuple[State, int] | None:
        """Feed `text` to the automaton.

        Returns:
            (new_state, number_of_chars_that_belong_to_the_value), or None if
            some char is rejected before the value is done.
        """
        curr = state
        consumed = 0

        for char in text:
            if self.is_done(curr):
                break

            next_state = self.step(curr, char)
            if next_state is None:
                if self.can_stop(curr):
                    break
                return None

            curr = next_state
            consumed += 1

        return curr, consumed


class NumberAutomaton(Automaton):
    """-?(0|[1-9][0-9]*) with an optional fraction (.[0-9]+) if not integer.

    No exponent and no leading zeros. A terminator is accepted only when the
    text so far is a complete number.
    """

    TERMINATORS = {",", "}", " ", "\t", "\n", "\r"}

    def __init__(self, integer: bool = False) -> None:
        self.integer = integer

    @property
    def key(self) -> Hashable:
        return ("number", self.integer)

    def start(self) -> State:
        return "START"

    def step(self, state: State, char: str) -> State | None:
        st = str(state)

        if st == "DONE":
            return None

        if st == "START":
            if char == "-":
                return "MINUS"
            if char == "0":
                return "ZERO"
            if char in "123456789":
                return "INT_DIGITS"
            return None

        if st == "MINUS":
            if char == "0":
                return "ZERO"
            if char in "123456789":
                return "INT_DIGITS"
            return None

        if st == "ZERO":
            if not self.integer and char == ".":
                return "DOT"
            if char in self.TERMINATORS:
                return "DONE"
            return None

        if st == "INT_DIGITS":
            if char.isdigit():
                return "INT_DIGITS"
            if not self.integer and char == ".":
                return "DOT"
            if char in self.TERMINATORS:
                return "DONE"
            return None

        if st == "DOT":
            if char.isdigit():
                return "FRAC_DIGITS"
            return None

        if st == "FRAC_DIGITS":
            if char.isdigit():
                return "FRAC_DIGITS"
            if char in self.TERMINATORS:
                return "DONE"
            return None

        return None

    def is_done(self, state: State) -> bool:
        return state == "DONE"

    def can_stop(self, state: State) -> bool:
        return state in {"ZERO", "INT_DIGITS", "FRAC_DIGITS", "DONE"}

    def shortest_completion(self, state: State) -> str:
        st = str(state)
        if st in {"ZERO", "INT_DIGITS", "FRAC_DIGITS", "DONE"}:
            return ""
        if st in {"START", "MINUS"}:
            return "0"
        if st == "DOT":
            return "0"
        return ""

    def consume(self, state: State, text: str) -> tuple[State, int] | None:
        curr = state
        consumed = 0

        for char in text:
            if self.is_done(curr):
                break

            next_state = self.step(curr, char)
            if next_state is None:
                if self.can_stop(curr):
                    break
                return None

            if next_state == "DONE":
                curr = next_state
                break

            curr = next_state
            consumed += 1

        return curr, consumed


class StringAutomaton(Automaton):
    """Body of a JSON string, starting AFTER the opening quote.

    Any char is allowed except control characters. A backslash must be followed
    by a valid escape (\\\\ \\" \\/ \\b \\f \\n \\r \\t \\uXXXX). The value is done
    once the closing quote is consumed.
    """

    def __init__(self) -> None:
        pass

    @property
    def key(self) -> Hashable:
        return "string"

    def start(self) -> State:
        return "NORMAL"

    def step(self, state: State, char: str) -> State | None:
        st = str(state)

        if st == "DONE":
            return None

        if st == "NORMAL":
            if char == '"':
                return "DONE"
            if char == "\\":
                return "ESCAPE"
            if ord(char) < 0x20:
                return None
            return "NORMAL"

        if st == "ESCAPE":
            if char in {'"', "\\", "/", "b", "f", "n", "r", "t"}:
                return "NORMAL"
            if char == "u":
                return "U0"
            return None

        if st in {"U0", "U1", "U2"}:
            if char in "0123456789abcdefABCDEF":
                idx = int(st[1]) + 1
                return "NORMAL" if idx == 4 else f"U{idx}"
            return None

        if st == "U3":
            if char in "0123456789abcdefABCDEF":
                return "NORMAL"
            return None

        return None

    def is_done(self, state: State) -> bool:
        return state == "DONE"

    def shortest_completion(self, state: State) -> str:
        st = str(state)
        if st == "DONE":
            return ""
        if st == "NORMAL":
            return '"'
        if st == "ESCAPE":
            return 'n"'
        if st == "U0":
            return '0000"'
        if st == "U1":
            return '000"'
        if st == "U2":
            return '00"'
        if st == "U3":
            return '0"'
        return '"'


class ChoiceAutomaton(Automaton):
    """Closed choice among literal texts."""

    def __init__(self, choices: Sequence[str]) -> None:
        self.choices = tuple(choices)

    @property
    def key(self) -> Hashable:
        return ("choice", self.choices)

    def start(self) -> State:
        return ""

    def step(self, state: State, char: str) -> State | None:
        prefix = str(state) + char
        if any(c.startswith(prefix) for c in self.choices):
            return prefix
        return None

    def is_done(self, state: State) -> bool:
        return str(state) in self.choices

    def forced_text(self, state: State) -> str | None:
        prefix = str(state)
        matching = [c for c in self.choices if c.startswith(prefix)]
        if len(matching) == 1:
            full = matching[0]
            if full != prefix:
                return full[len(prefix) :]
        return None

    def shortest_completion(self, state: State) -> str:
        prefix = str(state)
        matching = [c for c in self.choices if c.startswith(prefix)]
        if not matching:
            return ""
        best = min(matching, key=len)
        return best[len(prefix) :]


def automaton_for(spec: ParameterSpec) -> Automaton:
    """Pick the automaton that matches a declared parameter type."""
    if spec.enum is not None:
        formatted_choices = [json.dumps(val) for val in spec.enum]
        return ChoiceAutomaton(formatted_choices)

    t = spec.type
    if t == "number":
        return NumberAutomaton(integer=False)
    if t == "integer":
        return NumberAutomaton(integer=True)
    if t == "string":
        return StringAutomaton()
    if t == "boolean":
        return ChoiceAutomaton(["true", "false"])
    if t == "null":
        return ChoiceAutomaton(["null"])

    raise ValueError(f"Unsupported parameter type: {t}")


class MaskCache:
    """Builds and caches the allowed-token masks."""

    def __init__(self, vocab: Vocabulary) -> None:
        self.vocab = vocab
        self._cache: dict[tuple[Hashable, State], NDArray[np.bool_]] = {}

    def mask(self, automaton: Automaton, state: State) -> NDArray[np.bool_]:
        """Boolean mask over the vocabulary: True = allowed from this state."""
        cache_key = (automaton.key, state)
        if cache_key in self._cache:
            return self._cache[cache_key]

        mask = np.zeros(self.vocab.size, dtype=bool)

        for token_id, token_text in self.vocab.items():
            if not token_text:
                continue
            res = automaton.consume(state, token_text)
            if res is not None:
                mask[token_id] = True

        self._cache[cache_key] = mask
        return mask

    def __len__(self) -> int:
        return len(self._cache)
