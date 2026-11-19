"""Command-line interface for the project."""

import click

from src.preprocessing import (
    remove_missing,
    fill_missing,
    remove_duplicates,
    min_max_normalize,
    z_score_standardize,
    clip_values,
    to_integers,
    log_transform,
    tokenize,
    keep_alnum_and_spaces,
    remove_stop_words,
    flatten,
    shuffle_with_seed,
)


def _show(items):
    """Print a list as space-separated values."""
    click.echo(" ".join(str(item) for item in items))


@click.group()
def cli():
    """Data preprocessing command line tool."""


@cli.group()
def clean():
    """Commands for data cleaning."""


@cli.group()
def numeric():
    """Commands for dealing with numerical attributes."""


@cli.group()
def text():
    """Commands for dealing with textual information."""


@cli.group()
def struct():
    """Commands for the structure of data."""


# ---------- clean ----------


@clean.command(name="remove-missing")
@click.argument("values", nargs=-1)
def remove_missing_cmd(values):
    """Remove missing values (None, empty strings, nan) from a list.

    \b
    Example:
      python -m src.cli clean remove-missing 1 "" 2 None 3
    """
    _show(remove_missing(list(values)))


@clean.command(name="fill-missing")
@click.argument("values", nargs=-1)
@click.option(
    "--value",
    default="0",
    show_default=True,
    help="Value used to replace the missing values.",
)
def fill_missing_cmd(values, value):
    """Replace missing values with a given value.

    \b
    Example:
      python -m src.cli clean fill-missing 1 "" 2 --value 9
    """
    _show(fill_missing(list(values), value))


# ---------- numeric ----------


@numeric.command(name="normalize")
@click.argument("values", nargs=-1, type=float)
@click.option(
    "--new-min",
    default=0.0,
    show_default=True,
    type=float,
    help="Minimum of the new range.",
)
@click.option(
    "--new-max",
    default=1.0,
    show_default=True,
    type=float,
    help="Maximum of the new range.",
)
def normalize_cmd(values, new_min, new_max):
    """Min-max normalize numbers into a new range.

    \b
    Example:
      python -m src.cli numeric normalize 1 2 3 --new-min 0 --new-max 10
    """
    _show(min_max_normalize(list(values), new_min, new_max))


@numeric.command(name="standardize")
@click.argument("values", nargs=-1, type=float)
def standardize_cmd(values):
    """Standardize numbers using the z-score method.

    \b
    Example:
      python -m src.cli numeric standardize 1 2 3
    """
    _show(z_score_standardize(list(values)))


@numeric.command(name="clip")
@click.argument("values", nargs=-1, type=float)
@click.option(
    "--min-value",
    default=0.0,
    show_default=True,
    type=float,
    help="Minimum value to clip to.",
)
@click.option(
    "--max-value",
    default=1.0,
    show_default=True,
    type=float,
    help="Maximum value to clip to.",
)
def clip_cmd(values, min_value, max_value):
    """Clip numbers to a range.

    \b
    Example:
      python -m src.cli numeric clip -5 0.5 7 --min-value 0 --max-value 1
    """
    _show(clip_values(list(values), min_value, max_value))


@numeric.command(name="to-int")
@click.argument("values", nargs=-1)
def to_int_cmd(values):
    """Convert values to integers, dropping non-numeric ones.

    \b
    Example:
      python -m src.cli numeric to-int 1 2.5 abc 4
    """
    _show(to_integers(list(values)))


@numeric.command(name="log")
@click.argument("values", nargs=-1, type=float)
def log_cmd(values):
    """Apply a logarithmic transform to the positive numbers.

    \b
    Example:
      python -m src.cli numeric log 1 10 -3
    """
    _show(log_transform(list(values)))


# ---------- text ----------


@text.command(name="tokenize")
@click.argument("content")
def tokenize_cmd(content):
    """Split text into lowercase alphanumeric words.

    \b
    Example:
      python -m src.cli text tokenize "Hello, World!"
    """
    _show(tokenize(content))


@text.command(name="remove-punctuation")
@click.argument("content")
def remove_punctuation_cmd(content):
    """Keep only alphanumeric characters and spaces.

    \b
    Example:
      python -m src.cli text remove-punctuation "Hello, World!"
    """
    click.echo(keep_alnum_and_spaces(content))


@text.command(name="remove-stopwords")
@click.argument("content")
@click.option(
    "--stopword",
    "stopwords",
    multiple=True,
    help="A stop-word to remove. Repeat the option for more words.",
)
def remove_stopwords_cmd(content, stopwords):
    """Remove the given stop-words from the text (lowercased).

    \b
    Example:
      python -m src.cli text remove-stopwords "The cat is here" --stopword the --stopword is
    """
    click.echo(remove_stop_words(content, list(stopwords)))


# ---------- struct ----------


@struct.command(name="shuffle")
@click.argument("values", nargs=-1)
@click.option(
    "--seed", default=None, type=int, help="Integer seed for reproducible shuffling."
)
def shuffle_cmd(values, seed):
    """Shuffle a list of values.

    \b
    Example:
      python -m src.cli struct shuffle 1 2 3 4 --seed 42
    """
    _show(shuffle_with_seed(list(values), seed))


@struct.command(name="flatten")
@click.argument("lists", nargs=-1)
def flatten_cmd(lists):
    """Flatten a list of lists. Each argument is one inner list, with
    comma-separated values.

    \b
    Example:
      python -m src.cli struct flatten 1,2 3,4 5
    """
    _show(flatten([item.split(",") for item in lists]))


@struct.command(name="unique")
@click.argument("values", nargs=-1)
def unique_cmd(values):
    """Return the unique values of a list.

    \b
    Example:
      python -m src.cli struct unique 1 2 2 3 3
    """
    _show(remove_duplicates(list(values)))


if __name__ == "__main__":
    cli()
