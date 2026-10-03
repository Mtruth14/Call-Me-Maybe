from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

ParameterType = Literal["number", "integer", "string", "boolean", "null"]


class InputError(Exception):
    """An input file is invalid.

    The CLI prints the message on stderr and exits with code 1.
    """


class ParameterSpec(BaseModel):
    """Declaration of one parameter, e.g. {"type": "number"}."""

    type: ParameterType
    enum: list[Any] | None = None


class FunctionDefinition(BaseModel):
    """One entry of functions_definition.json.

    `parameters` keeps the order of the file: the decoder writes the arguments
    in that order.
    """

    name: str
    description: str = ""
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: dict[str, Any] = Field(default_factory=dict)

    def check_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate the generated arguments against the declaration.

        Checks that the keys are exactly the declared ones and that every value
        has the declared type. Values of type `number` are converted to float
        (40 -> 40.0). Returns the cleaned dict.

        Raises:
            ValueError: if a key is missing or unexpected, or a type is wrong.
        """
        expected_keys = set(self.parameters.keys())
        provided_keys = set(arguments.keys())

        if expected_keys != provided_keys:
            missing = expected_keys - provided_keys
            extra = provided_keys - expected_keys
            msg_parts = []
            if missing:
                msg_parts.append(f"missing key(s): {sorted(missing)}")
            if extra:
                msg_parts.append(f"unexpected key(s): {sorted(extra)}")
            raise ValueError(f"Argument mismatch for '{self.name}': {', '.join(msg_parts)}")

        cleaned_args: dict[str, Any] = {}

        for key, spec in self.parameters.items():
            val = arguments[key]
            expected_type = spec.type

            if spec.enum is not None and val not in spec.enum:
                raise ValueError(f"Value '{val}' for parameter '{key}' not in enum: {spec.enum}")

            if expected_type == "number":
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise ValueError(f"Parameter '{key}' must be a number, got {type(val).__name__}")
                cleaned_args[key] = float(val)

            elif expected_type == "integer":
                if isinstance(val, bool) or not isinstance(val, int):
                    raise ValueError(f"Parameter '{key}' must be an integer, got {type(val).__name__}")
                cleaned_args[key] = val

            elif expected_type == "string":
                if not isinstance(val, str):
                    raise ValueError(f"Parameter '{key}' must be a string, got {type(val).__name__}")
                cleaned_args[key] = val

            elif expected_type == "boolean":
                if not isinstance(val, bool):
                    raise ValueError(f"Parameter '{key}' must be a boolean, got {type(val).__name__}")
                cleaned_args[key] = val

            elif expected_type == "null":
                if val is not None:
                    raise ValueError(f"Parameter '{key}' must be null, got {type(val).__name__}")
                cleaned_args[key] = None

        return cleaned_args


class PromptItem(BaseModel):
    """One entry of function_calling_tests.json."""

    prompt: str


class FunctionCall(BaseModel):
    """One entry of the output file."""

    prompt: str
    name: str
    parameters: dict[str, Any]


def parse_functions(raw: Any) -> list[FunctionDefinition]:
    """Validate the content of functions_definition.json.

    Raises:
        InputError: not a list, invalid entry, unsupported parameter type,
            or two functions with the same name. The message is one line.
    """
    if not isinstance(raw, list):
        raise InputError("functions_definition.json must contain a JSON array.")

    seen_names: set[str] = set()
    functions: list[FunctionDefinition] = []

    for index, item in enumerate(raw):
        try:
            func = FunctionDefinition.model_validate(item)
        except ValidationError as err:
            raise InputError(
                f"Invalid function definition at index {index}: {err.errors()[0]['msg']}"
            ) from err

        if func.name in seen_names:
            raise InputError(f"Duplicate function name encountered: '{func.name}'")

        seen_names.add(func.name)
        functions.append(func)

    return functions


def parse_prompts(raw: Any) -> list[PromptItem]:
    """Validate the content of function_calling_tests.json.

    Raises:
        InputError: not a list, or an entry without a string "prompt".
    """
    if not isinstance(raw, list):
        raise InputError("function_calling_tests.json must contain a JSON array.")

    prompts: list[PromptItem] = []

    for index, item in enumerate(raw):
        try:
            prompt_obj = PromptItem.model_validate(item)
        except ValidationError as err:
            raise InputError(
                f"Invalid prompt item at index {index}: missing or non-string 'prompt'."
            ) from err

        prompts.append(prompt_obj)

    return prompts
