# MLops_Labs
Repo for the labs from the MLops subject.

The project is a small command-line tool for data preprocessing. The logic lives in
`src/preprocessing.py` and the CLI (built with Click) in `src/cli.py`.

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)

## Installation

From the root of the repository, install the dependencies:

```bash
uv sync
```

<!-- ## Running the CLI

Run the commands from the root of the repository:

```bash
uv run python -m src.cli --help
```

The commands are organized in four groups:

- `clean`: `remove-missing`, `fill-missing`
- `numeric`: `normalize`, `standardize`, `clip`, `to-int`, `log`
- `text`: `tokenize`, `remove-punctuation`, `remove-stopwords`
- `struct`: `shuffle`, `flatten`, `unique`

Use `--help` on any group or command to see its options, for example:

```bash
uv run python -m src.cli numeric normalize --help
```

Some examples:

```bash
uv run python -m src.cli clean fill-missing 1 "" 2 --value 9
uv run python -m src.cli numeric normalize 1 2 3 --new-min 0 --new-max 10
uv run python -m src.cli text tokenize "Hello, World!"
uv run python -m src.cli struct shuffle 1 2 3 4 --seed 42
```

To pass negative numbers, put `--` after the options so they are not read as options:

```bash
uv run python -m src.cli numeric clip --min-value 0 --max-value 1 -- -5 0.5 7
``` -->

## Running the tests

The tests are in the `tests` folder:

- `tests/test_logic.py`: unit tests for the functions in `src/preprocessing.py`
- `tests/test_cli.py`: tests for the CLI commands

Run all the tests from the root of the repository:

```bash
uv run python -m pytest -v
```

Run a single test file:

```bash
uv run python -m pytest tests/test_logic.py
```

Run the tests with a coverage report:

```bash
uv run python -m pytest --cov=src
```

Note: use `python -m pytest` instead of just `pytest`, so that the `src` package can be
imported from the tests.

## Code formatting & style

The code is formatted with Black and checked with Pylint:

```bash
uv run python -m pylint src/*.py
uv run black src/*.py
```
