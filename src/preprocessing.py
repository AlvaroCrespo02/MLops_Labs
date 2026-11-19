"""Preprocessing functions"""

import math
import random
import re


def _is_missing(value):
    """
    Return True if a value counts as missing.

    Missing values are:
      - None
      - '' (empty string)
      - NaN (float('nan'), numpy.nan, decimal NaN, etc.)
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, float):
        return math.isnan(value)
    return False


def remove_missing(values):
    """
    Return a new list with missing values removed.

    Missing values are None, '' (empty string), and NaN.
    The order of the remaining values is preserved.
    """
    return [v for v in values if not _is_missing(v)]


def fill_missing(values, fill_value=0):
    """
    Return a new list where missing values are replaced with fill_value.

    Missing values are None, '' (empty string), and NaN.
    All other values are kept as they are, in the same order.
    """
    return [fill_value if _is_missing(v) else v for v in values]


def remove_duplicates(values):
    """
    Return a new list with duplicate values removed.

    The first occurrence of each value is kept, and the original
    order is preserved.
    """
    result = []
    seen_hashable = set()
    seen_unhashable = []

    for value in values:
        try:
            if value in seen_hashable:
                continue
            seen_hashable.add(value)
        except TypeError:
            # Unhashable values (e.g. lists, dicts) can't go in a set,
            # so compare them by equality instead.
            if value in seen_unhashable:
                continue
            seen_unhashable.append(value)
        result.append(value)

    return result


def min_max_normalize(values, new_min=0.0, new_max=1.0):
    """
    Rescale numerical values to the range [new_min, new_max]
    using min-max normalization.

    Formula: (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    If all values are equal (old_max == old_min), every value is mapped
    to new_min, since there is no range to scale.
    """
    if not values:
        return []

    old_min = min(values)
    old_max = max(values)
    old_range = old_max - old_min

    if old_range == 0:
        return [float(new_min) for _ in values]

    new_range = new_max - new_min
    return [(x - old_min) / old_range * new_range + new_min for x in values]


def z_score_standardize(values):
    """
    Standardize numerical values using the z-score: (x - mean) / std.

    Uses the population standard deviation (divides by n).
    If all values are equal (std == 0), every value is mapped to 0.0.
    """
    if not values:
        return []

    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(variance)

    if std == 0:
        return [0.0 for _ in values]

    return [(x - mean) / std for x in values]


def clip_values(values, min_value, max_value):
    """
    Clip each value to the range [min_value, max_value].

    Values below min_value become min_value, and values above
    max_value become max_value.
    """
    if min_value > max_value:
        raise ValueError("min_value must be less than or equal to max_value")

    return [min(max(x, min_value), max_value) for x in values]


def to_integers(values):
    """
    Convert lists of strings to integers.

    Numerical strings such as "42", " -7 " and "3.9" are converted.
    Decimal strings are truncated toward zero ("3.9" -> 3).
    Values that can't be parsed as numbers (e.g. "abc", "", "nan") are excluded.
    """
    result = []
    for value in values:
        try:
            result.append(int(value))
        except (ValueError, TypeError):
            try:
                number = float(value)
            except (ValueError, TypeError):
                continue
            # Skip nan and inf, which can't be converted to int
            if math.isnan(number) or math.isinf(number):
                continue
            result.append(int(number))
    return result


def log_transform(values):
    """
    Apply the natural logarithm to each positive numerical value.

    Zero, negative values, and non-numerical values are excluded,
    since their logarithm is undefined.
    """
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isnan(value) or value <= 0:
            continue
        result.append(math.log(value))
    return result


def tokenize(text):
    """
    Split text into lowercase words, keeping only alphanumeric characters.

    Any character that is not a letter or digit (punctuation, underscores,
    symbols, whitespace) separates words and is discarded.
    Returns a list of words.
    """
    return re.findall(r"[^\W_]+", text.lower())


def keep_alnum_and_spaces(text):
    """
    Remove all characters except letters, digits, and spaces.

    Punctuation, symbols, and underscores are removed.
    Letters and digits from any language are kept.
    """
    return re.sub(r"[^\w\s]|_", "", text)


def remove_stop_words(text, stop_words):
    """
    Lowercase the text and remove the given stop words.

    The stop words are compared in lowercase, so 'The' and 'the' both match.
    Words are joined with single spaces, and punctuation is dropped.
    """
    stops = {word.lower() for word in stop_words}
    return " ".join(word for word in tokenize(text) if word not in stops)


def flatten(list_of_lists):
    """
    Flatten a list of lists into a single list.

    Only one level is flattened, so elements that are themselves
    lists are kept as they are.
    """
    return [item for sublist in list_of_lists for item in sublist]


def shuffle_with_seed(values, seed):
    """
    Return a new list with the values in random order.

    The same seed always produces the same order. The input list is not modified.
    """
    rng = random.Random(seed)
    shuffled = list(values)
    rng.shuffle(shuffled)
    return shuffled
