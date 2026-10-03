"""Constrained generation of one function call.

The program writes every piece of syntax; the model only chooses
  1. the function name (closed choice among the known names),
  2. the value of each parameter (constrained by the declared type).

Layout of the generated text, as the model sees it:
    {"name":"fn_x","parameters":{"a": <value>, "b": <value>}}
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from src.automata import ChoiceAutomaton, MaskCache, State, automaton_for, Automaton
from src.models import FunctionCall, FunctionDefinition
from src.vocab import Vocabulary


class DecodingError(Exception):
    """Generation failed for one prompt (invalid value, budget exhausted...)."""


class _IdsTensor(Protocol):
    def tolist(self) -> list[list[int]]: ...


class LanguageModel(Protocol):
    """The public methods of llm_sdk.Small_LLM_Model that we use."""

    def encode(self, text: str) -> _IdsTensor: ...

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]: ...

    def get_path_to_vocab_file(self) -> str: ...


def build_selection_prompt(functions: list[FunctionDefinition], user_prompt: str) -> str:
    """Prompt used to choose the function."""
    funcs_doc = "\n".join(
        f"- {fn.name}: {fn.description}" for fn in functions
    )
    return (
        "Available functions:\n"
        f"{funcs_doc}\n\n"
        f"User request: {user_prompt}\n\n"
        'JSON output:\n{"name":"'
    )


def build_arguments_prompt(function: FunctionDefinition, user_prompt: str) -> str:
    """Shorter prompt used once the function is chosen."""
    params_doc = ", ".join(
        f"{k}: {v.type}" for k, v in function.parameters.items()
    )
    return (
        f"Function: {function.name}\n"
        f"Description: {function.description}\n"
        f"Parameters: {{{params_doc}}}\n\n"
        f"User request: {user_prompt}\n\n"
        "Extract argument values exact from the user request. Use standard regex for patterns.\n"
        f'JSON output:\n{{"name":"{function.name}","parameters":{{'
    )


class ConstrainedDecoder:
    """Turns a prompt into a validated `FunctionCall`."""

    def __init__(
        self,
        model: LanguageModel,
        vocab: Vocabulary,
        functions: list[FunctionDefinition],
        max_value_tokens: int = 64,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.functions = functions
        self.max_value_tokens = max_value_tokens
        self.cache = MaskCache(vocab)
        self._func_map = {f.name: f for f in functions}

    def decode(self, user_prompt: str) -> FunctionCall:
        """Main entry point: choose the function, then generate its arguments."""
        chosen_fn = self._choose_function(user_prompt)
        args = self._generate_arguments(chosen_fn, user_prompt)

        try:
            cleaned_args = chosen_fn.check_arguments(args)
        except ValueError as err:
            raise DecodingError(f"Argument check failed: {err}") from err

        return FunctionCall(
            prompt=user_prompt,
            name=chosen_fn.name,
            parameters=cleaned_args,
        )

    def _choose_function(self, user_prompt: str) -> FunctionDefinition:
        """Closed choice among the function names (ChoiceAutomaton)."""
        prompt = build_selection_prompt(self.functions, user_prompt)
        tensor = self.model.encode(prompt)
        ids = list(tensor.tolist()[0])

        fn_names = [f.name for f in self.functions]
        automaton = ChoiceAutomaton(fn_names)

        raw_name = self._generate_value(ids, automaton)
        if raw_name not in self._func_map:
            raise DecodingError(f"Selected function '{raw_name}' is not registered.")

        return self._func_map[raw_name]

    def _generate_arguments(
        self, function: FunctionDefinition, user_prompt: str
    ) -> dict[str, Any]:
        """Loop over the declared parameters, in schema order."""
        prompt = build_arguments_prompt(function, user_prompt)
        tensor = self.model.encode(prompt)
        ids = list(tensor.tolist()[0])

        result: dict[str, Any] = {}
        param_items = list(function.parameters.items())

        for idx, (param_name, param_spec) in enumerate(param_items):
            key_syntax = f'"{param_name}":'
            self._append_text(ids, key_syntax)

            if param_spec.type == "string" and param_spec.enum is None:
                self._append_text(ids, '"')
                automaton = automaton_for(param_spec)
                raw_val = self._generate_value(ids, automaton)
                if raw_val.endswith('"'):
                    raw_val = raw_val[:-1]
                parsed_val = raw_val
            else:
                automaton = automaton_for(param_spec)
                raw_val = self._generate_value(ids, automaton)
                parsed_val = self._parse_value(raw_val, param_spec.type)

            result[param_name] = parsed_val

            is_last = idx == len(param_items) - 1
            separator = "}}" if is_last else ","
            self._append_text(ids, separator)

        return result

    def _generate_value(self, ids: list[int], automaton: Automaton) -> str:
        """Core loop. Returns the raw text of one value."""
        state: State = automaton.start()
        value_text = ""
        tokens_used = 0

        while not automaton.is_done(state) and tokens_used < self.max_value_tokens:
            forced = automaton.forced_text(state)
            if forced is not None:
                value_text += forced
                self._append_text(ids, forced)
                res = automaton.consume(state, forced)
                if res is None:
                    raise DecodingError("Forced text rejected by automaton state.")
                state, _ = res
                continue

            mask = self.cache.mask(automaton, state)
            next_id = self._next_token(ids, mask)
            ids.append(next_id)
            tokens_used += 1

            token_str = self.vocab.text(next_id)
            res = automaton.consume(state, token_str)
            if res is None:
                raise DecodingError(
                    f"Token '{token_str}' invalid for state {state}."
                )

            state, num_consumed = res
            value_text += token_str[:num_consumed]

        if not automaton.is_done(state):
            completion = automaton.shortest_completion(state)
            if completion:
                value_text += completion
                self._append_text(ids, completion)

        return value_text

    def _next_token(self, ids: list[int], mask: NDArray[np.bool_]) -> int:
        """Logits of the next token, forbidden ones set to -inf, then argmax."""
        if not np.any(mask):
            raise DecodingError("No allowed token left in the vocabulary mask.")

        raw_logits = self.model.get_logits_from_input_ids(ids)
        logits_arr = np.array(raw_logits, dtype=np.float32)

        if len(logits_arr) < len(mask):
            padded_logits = np.full(len(mask), -np.inf, dtype=np.float32)
            padded_logits[: len(logits_arr)] = logits_arr
            logits_arr = padded_logits
        elif len(logits_arr) > len(mask):
            logits_arr = logits_arr[: len(mask)]

        logits_arr[~mask] = -np.inf
        return int(np.argmax(logits_arr))

    def _append_text(self, ids: list[int], text: str) -> None:
        """Encode `text` and append its token ids to `ids`."""
        encoded_tensor = self.model.encode(text)
        text_ids = encoded_tensor.tolist()[0]
        ids.extend(text_ids)

    @staticmethod
    def _parse_value(raw: str, parameter_type: str) -> Any:
        """json.loads(raw) with a type check."""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as err:
            raise DecodingError(f"Failed to parse generated value JSON '{raw}': {err}") from err

        if parameter_type == "number":
            if isinstance(parsed, bool) or not isinstance(parsed, (int, float)):
                raise DecodingError(f"Expected number, got {type(parsed).__name__}")
            return float(parsed)

        if parameter_type == "integer":
            if isinstance(parsed, bool) or not isinstance(parsed, int):
                raise DecodingError(f"Expected integer, got {type(parsed).__name__}")
            return parsed

        if parameter_type == "string":
            if not isinstance(parsed, str):
                raise DecodingError(f"Expected string, got {type(parsed).__name__}")
            return parsed

        if parameter_type == "boolean":
            if not isinstance(parsed, bool):
                raise DecodingError(f"Expected boolean, got {type(parsed).__name__}")
            return parsed

        if parameter_type == "null":
            if parsed is not None:
                raise DecodingError(f"Expected null, got {type(parsed).__name__}")
            return None

        return parsed
