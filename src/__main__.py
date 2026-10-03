from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.decoder import ConstrainedDecoder, DecodingError
from src.models import (
    FunctionCall,
    FunctionDefinition,
    InputError,
    PromptItem,
    parse_functions,
    parse_prompts,
)
from src.vocab import Vocabulary


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Options: --functions_definition, --input, --output (with the default paths of
    the subject) and an optional --model (default: Qwen/Qwen3-0.6B).
    """
    parser = argparse.ArgumentParser(
        description="Translate natural language prompts to structured function calls."
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="Path to functions definition JSON file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="Path to input test prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Path to output results JSON file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model identifier or name.",
    )
    return parser.parse_args(argv)


def read_json(path: Path) -> Any:
    """Read and parse a JSON file.

    Raises:
        InputError: file missing, unreadable, or not valid JSON (one-line message).
    """
    if not path.is_file():
        raise InputError(f"File not found or not a regular file: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as err:
        raise InputError(f"Failed to read JSON from '{path}': {err}") from err


def load_inputs(
    functions_path: Path, prompts_path: Path
) -> tuple[list[FunctionDefinition], list[PromptItem]]:
    """Read both input files and validate them (`parse_functions`, `parse_prompts`)."""
    raw_functions = read_json(functions_path)
    functions = parse_functions(raw_functions)

    raw_prompts = read_json(prompts_path)
    prompts = parse_prompts(raw_prompts)

    return functions, prompts


def build_decoder(
    model_name: str, functions: list[FunctionDefinition]
) -> ConstrainedDecoder:
    """Load the model, build the vocabulary and the decoder.

    Raises:
        InputError: the model cannot be loaded (one-line message).
    """
    try:
        from llm_sdk import Small_LLM_Model  # type: ignore
    except ImportError as err:
        raise InputError(f"Failed to import llm_sdk: {err}") from err

    try:
        model = Small_LLM_Model(model_name)
    except Exception as err:
        raise InputError(f"Failed to initialize model '{model_name}': {err}") from err

    try:
        vocab_path = model.get_path_to_vocab_file()
        dummy_prompt = "test"
        dummy_ids = model.encode(dummy_prompt).tolist()[0]
        logits = model.get_logits_from_input_ids(dummy_ids)
        vocab_size = len(logits)

        vocab = Vocabulary.from_file(vocab_path, size=vocab_size)
    except Exception as err:
        raise InputError(f"Failed to build vocabulary: {err}") from err

    return ConstrainedDecoder(
        model=model,
        vocab=vocab,
        functions=functions,
    )


def process_prompts(
    decoder: ConstrainedDecoder, prompts: list[PromptItem]
) -> list[FunctionCall]:
    """Run the decoder on every prompt, in order.

    On DecodingError: write the reason on stderr and continue with the next
    prompt. The policy for a failed prompt (skip it or write a marked entry)
    must follow the subject.
    """
    results: list[FunctionCall] = []

    for item in prompts:
        try:
            call = decoder.decode(item.prompt)
            results.append(call)
        except DecodingError as err:
            sys.stderr.write(f"Decoding error for prompt '{item.prompt}': {err}\n")

    return results


def write_results(path: Path, results: list[FunctionCall]) -> None:
    """Create the parent directory if needed and write the JSON array."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        dumpable_data = [item.model_dump() for item in results]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dumpable_data, f, indent=2, ensure_ascii=False)
    except OSError as err:
        raise InputError(f"Failed to write results to '{path}': {err}") from err


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline and return the exit code (0 = ok, 1 = input error).

    Prints the number of prompts and the total time at the end.
    """
    start_time = time.perf_counter()
    args = parse_arguments(argv)

    try:
        functions, prompts = load_inputs(args.functions_definition, args.input)
        decoder = build_decoder(args.model, functions)
        results = process_prompts(decoder, prompts)
        write_results(args.output, results)
    except InputError as err:
        sys.stderr.write(f"Error: {err}\n")
        return 1

    elapsed = time.perf_counter() - start_time
    print(
        f"Processed {len(prompts)} prompt(s) successfully in {elapsed:.2f} seconds."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
