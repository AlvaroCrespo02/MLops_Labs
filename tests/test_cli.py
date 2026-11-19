# tests/test_cli_integration.py
import math

import pytest

from click.testing import CliRunner
from src.cli import cli
from src.preprocessing import shuffle_with_seed

# ---------- fixtures ----------

@pytest.fixture
def runner():
    return CliRunner()


# ---------- helpers ----------

def _floats(output):
    """Turn the CLI's space-separated output into a list of floats."""
    return [float(v) for v in output.split()]


# ---------- root group ----------

def test_root_help_lists_all_groups(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for group in ["clean", "numeric", "text", "struct"]:
        assert group in result.output


# ---------- clean group ----------

def test_clean_remove_missing(runner):
    result = runner.invoke(cli, ["clean", "remove-missing", "1", "", "2", "", "3"])
    assert result.exit_code == 0
    assert result.output.split() == ["1", "2", "3"]


@pytest.mark.parametrize(
    "args, expected",
    [
        (["1", "", "2"], ["1", "0", "2"]),                   # default --value 0
        (["1", "", "2", "--value", "9"], ["1", "9", "2"]),   # option given
        (["", "", "x"], ["0", "0", "x"]),
    ],
)
def test_clean_fill_missing(runner, args, expected):
    result = runner.invoke(cli, ["clean", "fill-missing", *args])
    assert result.exit_code == 0
    assert result.output.split() == expected


# ---------- numeric group ----------

@pytest.mark.parametrize(
    "options, expected",
    [
        ([], [0.0, 0.5, 1.0]),                                      # defaults
        (["--new-min", "0", "--new-max", "10"], [0.0, 5.0, 10.0]),
        (["--new-min", "-1", "--new-max", "1"], [-1.0, 0.0, 1.0]),
    ],
)
def test_numeric_normalize(runner, options, expected):
    result = runner.invoke(cli, ["numeric", "normalize", "1", "2", "3", *options])
    assert result.exit_code == 0
    assert _floats(result.output) == pytest.approx(expected)


def test_numeric_standardize(runner):
    result = runner.invoke(cli, ["numeric", "standardize", "2", "4", "4", "4", "5", "5", "7", "9"])
    assert result.exit_code == 0
    assert _floats(result.output) == pytest.approx(
        [-1.5, -0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 2.0]
    )


def test_numeric_clip_with_options(runner):
    # "--" ends option parsing, so negative values are read as arguments
    args = ["numeric", "clip", "--min-value", "0", "--max-value", "1", "--", "-5", "0.5", "7"]
    result = runner.invoke(cli, args)
    assert result.exit_code == 0
    assert _floats(result.output) == pytest.approx([0.0, 0.5, 1.0])


def test_numeric_clip_invalid_range_fails(runner):
    args = ["numeric", "clip", "--min-value", "5", "--max-value", "1", "--", "1", "2"]
    result = runner.invoke(cli, args)
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)


def test_numeric_to_int(runner):
    result = runner.invoke(cli, ["numeric", "to-int", "1", "2.5", "abc", "4"])
    assert result.exit_code == 0
    assert result.output.split() == ["1", "2", "4"]


@pytest.mark.parametrize(
    "values, expected",
    [
        (["1", "10"], [0.0, math.log(10)]),
        (["1", "10", "0"], [0.0, math.log(10)]),   # zero is excluded
        (["0.5"], [math.log(0.5)]),
    ],
)
def test_numeric_log(runner, values, expected):
    result = runner.invoke(cli, ["numeric", "log", *values])
    assert result.exit_code == 0
    assert _floats(result.output) == pytest.approx(expected)


def test_numeric_rejects_non_number(runner):
    result = runner.invoke(cli, ["numeric", "normalize", "abc"])
    assert result.exit_code == 2   # Click usage error


# ---------- text group ----------

def test_text_tokenize(runner):
    result = runner.invoke(cli, ["text", "tokenize", "Hello, World!"])
    assert result.exit_code == 0
    assert result.output.strip() == "hello world"


def test_text_remove_punctuation(runner):
    result = runner.invoke(cli, ["text", "remove-punctuation", "Hello, World! 123_ok"])
    assert result.exit_code == 0
    assert result.output.strip() == "Hello World 123ok"


@pytest.mark.parametrize(
    "args, expected",
    [
        (["The cat is here", "--stopword", "the", "--stopword", "is"], "cat here"),
        (["The cat is here", "--stopword", "THE"], "cat is here"),
        (["The cat is here"], "the cat is here"),   # no stop-words given
    ],
)
def test_text_remove_stopwords(runner, args, expected):
    result = runner.invoke(cli, ["text", "remove-stopwords", *args])
    assert result.exit_code == 0
    assert result.output.strip() == expected


# ---------- struct group ----------

@pytest.mark.parametrize(
    "values, seed",
    [
        (["1", "2", "3", "4"], 42),
        (["a", "b", "c"], 0),
    ],
)
def test_struct_shuffle_with_seed_matches_logic(runner, values, seed):
    result = runner.invoke(cli, ["struct", "shuffle", *values, "--seed", str(seed)])
    assert result.exit_code == 0
    assert result.output.split() == shuffle_with_seed(values, seed)


def test_struct_shuffle_without_seed_keeps_all_values(runner):
    values = ["1", "2", "3", "4", "5"]
    result = runner.invoke(cli, ["struct", "shuffle", *values])
    assert result.exit_code == 0
    assert sorted(result.output.split()) == sorted(values)


def test_struct_flatten(runner):
    result = runner.invoke(cli, ["struct", "flatten", "1,2", "3,4", "5"])
    assert result.exit_code == 0
    assert result.output.split() == ["1", "2", "3", "4", "5"]


def test_struct_unique(runner):
    result = runner.invoke(cli, ["struct", "unique", "1", "2", "2", "3", "3"])
    assert result.exit_code == 0
    assert result.output.split() == ["1", "2", "3"]


# ---------- unknown command ----------

def test_unknown_group_fails(runner):
    result = runner.invoke(cli, ["nonexistent", "command"])
    assert result.exit_code == 2