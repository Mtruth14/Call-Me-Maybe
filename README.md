*This project has been created as part of the 42 curriculum by imoundhi.*

# Call Me Maybe

## Description

Call Me Maybe is a function-calling tool. It reads a natural-language request and
turns it into a structured function call. It does **not** answer the request.

> "What is the sum of 40 and 2?"
> → `{"prompt": "What is the sum of 40 and 2?", "name": "fn_add_numbers", "parameters": {"a": 40.0, "b": 2.0}}`

The program runs `Qwen/Qwen3-0.6B` through the provided `llm_sdk`. A model this
small is unreliable when it writes JSON freely. The output is therefore controlled
by **constrained decoding**: at each step, the logits of the forbidden tokens are
set to `-inf` and the next token is the `argmax` of what is left. The result is
valid JSON that matches the function schema by construction.

## Instructions

Requirements: Python 3.10+ and [`uv`](https://docs.astral.sh/uv/).

```bash
make install      # uv sync
make run          # reads data/input/*, writes data/output/function_calling_results.json
make debug        # run under pdb
make lint         # flake8 + mypy (standard flags)
make lint-strict  # flake8 + mypy --strict
make clean        # remove caches
```

Custom paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

The same paths can be given to `make` (`make run INPUT=my_prompts.json`). An optional
`--model` argument selects another Hugging Face model (default: `Qwen/Qwen3-0.6B`).

**Input files**

- `functions_definition.json`: a list of `{name, description, parameters, returns}`.
  Supported parameter types: `number`, `integer`, `string`, `boolean`, `null`
  (and `enum`).
- `function_calling_tests.json`: a list of `{"prompt": "..."}`.

**Output file**: a JSON array of `{"prompt", "name", "parameters"}` objects, in the
order of the input prompts.

**Errors**: a missing file, invalid JSON or an invalid entry gives a one-line
message on `stderr` and exit code 1, never a traceback.

## Architecture & Algorithm

```
src/models.py     Pydantic models, validation of the input files and of the result
src/vocab.py      byte-level BPE vocabulary -> real text
src/automata.py   one automaton per value type, token masks and their cache
src/decoder.py    the generation loop, driven by the function schema
src/__main__.py   command line, reading, writing
llm_sdk/          provided SDK (encode, get_logits_from_input_ids, vocab path)
```

### 1. Vocabulary (`vocab.py`)

`vocab.json` does not contain plain text. Qwen uses a byte-level BPE: each byte is
stored as a printable character (GPT-2 `bytes_to_unicode` table), so `Ġ` is a space
and `é` is stored as two characters (its two UTF-8 bytes). Each entry is mapped back
to bytes, then decoded as UTF-8.

A token that is only a fragment of a multi-byte character cannot be decoded alone.
It is excluded. Ids that exist in the logits but not in `vocab.json` (special tokens)
have no text and are never allowed.

### 2. Automata (`automata.py`)

An automaton describes the text of **one value**, character by character.

| Automaton          | Used for                      | Accepted text                                         |
| ------------------ | ----------------------------- | ----------------------------------------------------- |
| `NumberAutomaton`  | `number`, `integer`           | `-?(0\|[1-9][0-9]*)`, plus `(\.[0-9]+)?` for `number` |
| `StringAutomaton`  | `string`                      | body of a JSON string: no control char, valid escapes |
| `ChoiceAutomaton`  | function name, `boolean`, `null`, `enum` | one of a closed list of literals           |

A token is allowed from a state if the automaton accepts all its characters in
order. If the value ends in the middle of a token, the rest is dropped (the token
`"}}` closes a string, and the `}}` is ignored because the program writes the syntax).

A number has no closing mark. It ends when the model prefers a token starting with a
terminator (`,`, `}` or whitespace) to another digit. The terminator is not part of
the value.

`MaskCache` turns an (automaton, state) pair into a boolean numpy mask over the
vocabulary. The first request scans the vocabulary, later requests are a dictionary
lookup.

### 3. Generation (`decoder.py`)

The program writes all the syntax. The model only chooses the **function name** and
the **value of each argument**.

```
{"name":"fn_x","parameters":{"a": <value>, "b": <value>}}
```

1. **Function name.** A `ChoiceAutomaton` over the known names. The model decides
   only at a real fork. When one name remains, the rest is appended with no model call.
2. **Arguments.** The parameters are taken **from the schema, in schema order**. For
   each one, the program writes `"key": `, the model generates the value under the
   automaton of the declared type, then the program writes `,` or `}}`.
3. **Value loop.** At each step: logits → mask of the current state → `argmax` →
   the automaton consumes the token. If only one continuation is possible, it is
   appended without a model call.
4. **Budget.** A value has its own token budget. If it is exhausted, the value is
   closed with the shortest valid completion (closing quote for a string, a valid
   number for a number).

Two prompts are used. The first one lists the functions (names and descriptions) and
is used to choose the name. The second one contains only the chosen function with its
typed parameters, which is shorter and easier for a 0.6B model.

### 4. Validation (`models.py`)

The result is validated against the function definition: the keys are exactly the
declared ones and each value has the declared type. Values of type `number` are
converted to `float` (`40` becomes `40.0`). Each output entry is a Pydantic model.

## Design decisions

- **Driven by the schema, not by the text.** The state is known from the position in
  the schema and from the automaton, so the generated text is never re-scanned with
  regular expressions.
- **No constrained-decoding library.** The automata, the masks and the loop are
  written by hand and use only the public methods of `llm_sdk` (`encode`,
  `get_logits_from_input_ids`, `get_path_to_vocab_file`).
- **The program writes the syntax, the model chooses the content.** Braces, quotes,
  keys and commas are never left to the model.
- **Small automata, cached masks.** There are few distinct states, so most masks are
  computed once.
- **Generic prompts.** They contain no example tied to a particular function, so they
  work with any function definition file.
- **A model interface instead of the SDK class.** The decoder depends on a small
  `LanguageModel` protocol, so it can be tested with a fake model, without loading
  Qwen.
- **No exponent notation** for numbers. It is never needed to read a value from a
  sentence, and it can overflow to `inf`, which is not valid JSON.
- **Errors are reported, not hidden.** A prompt that cannot be decoded is reported on
  `stderr` with the reason. Errors are reported, not hidden. A prompt that cannot be decoded is reported on
stderr with the reason, and skipped in the final JSON array (or represented with null parameters depending on the strict mode requirement).

## Performance analysis

- **Validity.** The JSON syntax, the function name, the keys and the types are correct
  by construction for the five supported types.
- **Accuracy.** The constraint guarantees the *form*, not the *meaning*. Choosing the
  right function and extracting the right values depends on the 0.6B model and on
  the prompt.
- **Speed.** One forward pass per model-chosen token. The SDK exposes no KV cache, so
  the whole sequence is recomputed at each step. Forced text and single-choice steps
  cost no forward pass. The total time is printed at the end of a run.
  Speed. One forward pass per model-chosen token. The SDK exposes no KV cache, so
the whole sequence is recomputed at each step. Forced text and single-choice steps
cost no forward pass. The total time on the 11 test prompts was ~55.76 seconds on 
the workstation machine (processed 11 prompts successfully).
- **Memory.** The model, plus one boolean mask per cached state.

## Known limitations

- Only `number`, `integer`, `string`, `boolean`, `null` and `enum` are supported.
  Arrays and nested objects are not.
- Greedy decoding (`argmax`): no sampling, no beam search.
- Tokens that are fragments of a multi-byte character are never generated, so a
  character can only be produced by a token that contains it entirely.
- No KV cache, so long prompts make each step slower.

## Challenges

- **Tokens bigger than the grammar.** A token like `"}}` closes a string and also
  contains syntax. Solved with the `consume` contract: keep what belongs to the value,
  drop the rest.
- **Byte-level BPE.** Vocabulary entries are not readable text. Solved by decoding
  them with the GPT-2 byte table and keeping only the tokens that decode alone.
- **Numbers have no end mark.** Solved by accepting a terminator token only when the
  text so far is a complete number.
- **Backslashes in strings.** A regular expression such as `\d+` must be written
  `\\d+` in JSON. The string automaton accepts only valid escapes, so the output is
  always parseable.
- **Small model.** It needs a short, focused prompt. Rebuilding the prompt around the
  single chosen function improved the extraction of arguments.

In addition: `make lint` and `make lint-strict`, and runs on the provided sample files
and on hand-made input files (missing file, invalid JSON, wrong shape, empty list).

## Resources

- Hugging Face documentation: tokenizers and byte-level BPE; Qwen3 model card.
- GPT-2 byte-to-unicode table (OpenAI).
- JSON grammar: [json.org](https://www.json.org).
- Articles on constrained / structured decoding (logit masking).
- Pydantic v2 and pytest documentation.

**Use of AI:** an AI assistant was used to review the code, to design the module
structure, and to help write this README. The code can be explained by its author.