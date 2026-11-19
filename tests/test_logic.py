"""Logic testing"""
import math
import pytest
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

# ---------- Fixtures ----------

@pytest.fixture
def numbers():
    return [1.0, 2.0, 3.0, 4.0, 5.0]

@pytest.fixture
def mixed_values():
    return [1, None, "", float("nan"), 2, "a"]

# ---------- remove_missing ----------

def test_remove_missing_with_fixture(mixed_values):
    assert remove_missing(mixed_values) == [1, 2, "a"]


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1, None, 2], [1, 2]),
        (["a", "", "b"], ["a", "b"]),
        ([], []),
        ([1.0, float("nan"), 2.0], [1.0, 2.0]),
        ([None, None], []),
    ],
)
def test_remove_missing(values, expected):
    assert remove_missing(values) == expected


# ---------- fill_missing ----------

def test_fill_missing_default_value():
    assert fill_missing([None, 2, ""]) == [0, 2, 0]


@pytest.mark.parametrize(
    "values, fill_value, expected",
    [
        ([1, None, ""], 0, [1, 0, 0]),
        (["a", None, float("nan")], "?", ["a", "?", "?"]),
        ([None], -1, [-1]),
        ([5, 6], 99, [5, 6]),
    ],
)
def test_fill_missing_with_options(values, fill_value, expected):
    assert fill_missing(values, fill_value) == expected


# ---------- remove_duplicates ----------

@pytest.mark.parametrize(
    "values, expected",
    [
        ([1, 2, 1, 3, 2], [1, 2, 3]),
        (["b", "a", "b"], ["b", "a"]),
        ([], []),
        ([[1], [2], [1]], [[1], [2]]),          # unhashable values
        ([{"a": 1}, {"a": 1}, {"b": 2}], [{"a": 1}, {"b": 2}]),
    ],
)
def test_remove_duplicates(values, expected):
    assert remove_duplicates(values) == expected


# ---------- min_max_normalize ----------

def test_normalize_default_range(numbers):
    result = min_max_normalize(numbers)
    assert result[0] == 0.0
    assert result[-1] == 1.0


def test_normalize_constant_values_map_to_new_min():
    assert min_max_normalize([7, 7, 7]) == [0.0, 0.0, 0.0]


def test_normalize_empty_list():
    assert min_max_normalize([]) == []


@pytest.mark.parametrize(
    "new_min, new_max, expected_last",
    [
        (0.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (0.0, 10.0, 10.0),
    ],
)
def test_normalize_with_options(numbers, new_min, new_max, expected_last):
    result = min_max_normalize(numbers, new_min, new_max)
    assert result[0] == new_min
    assert result[-1] == expected_last


@pytest.mark.parametrize(
    "values, new_min, new_max, expected",
    [
        ([0, 5, 10], 0, 1, [0.0, 0.5, 1.0]),
        ([0, 5, 10], -1, 1, [-1.0, 0.0, 1.0]),
        ([10, 20], 0, 100, [0.0, 100.0]),
    ],
)
def test_normalize_middle_values(values, new_min, new_max, expected):
    assert min_max_normalize(values, new_min, new_max) == pytest.approx(expected)


# ---------- z_score_standardize ----------

def test_z_score_with_fixture(numbers):
    result = z_score_standardize(numbers)
    expected = [-2 / math.sqrt(2), -1 / math.sqrt(2), 0.0,
                1 / math.sqrt(2), 2 / math.sqrt(2)]
    assert result == pytest.approx(expected)


def test_z_score_mean_zero_and_std_one(numbers):
    result = z_score_standardize(numbers)
    assert sum(result) / len(result) == pytest.approx(0.0)
    variance = sum(x ** 2 for x in result) / len(result)
    assert variance == pytest.approx(1.0)


def test_z_score_known_population_values():
    result = z_score_standardize([2, 4, 4, 4, 5, 5, 7, 9])
    assert result == pytest.approx([-1.5, -0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 2.0])


def test_z_score_constant_values():
    assert z_score_standardize([3, 3, 3]) == [0.0, 0.0, 0.0]


def test_z_score_empty_list():
    assert z_score_standardize([]) == []


# ---------- clip_values ----------

@pytest.mark.parametrize(
    "values, min_value, max_value, expected",
    [
        ([-5, 0, 5, 10], 0, 5, [0, 0, 5, 5]),
        ([1.5, 2.5], 2.0, 3.0, [2.0, 2.5]),
        ([], 0, 1, []),
        ([4, 4], 4, 4, [4, 4]),
    ],
)
def test_clip_values(values, min_value, max_value, expected):
    assert clip_values(values, min_value, max_value) == expected


def test_clip_values_invalid_range_raises():
    with pytest.raises(ValueError):
        clip_values([1, 2], 5, 1)


# ---------- to_integers ----------

@pytest.mark.parametrize(
    "values, expected",
    [
        (["42", " -7 ", "3.9", "-2.7"], [42, -7, 3, -2]),
        (["abc", "", "nan", "inf", None], []),
        (["10", "x", "20"], [10, 20]),
        ([], []),
    ],
)
def test_to_integers(values, expected):
    assert to_integers(values) == expected


# ---------- log_transform ----------

def test_log_transform_skips_invalid_values():
    result = log_transform([1, math.e, 0, -3, "a", True, float("nan")])
    assert result == pytest.approx([0.0, 1.0])


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1, 10], [0.0, math.log(10)]),
        ([0.5], [math.log(0.5)]),
        ([], []),
        ([0, -1], []),
    ],
)
def test_log_transform(values, expected):
    assert log_transform(values) == pytest.approx(expected)


# ---------- tokenize ----------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello, World!", ["hello", "world"]),
        ("snake_case_word", ["snake", "case", "word"]),
        ("", []),
        ("Ünïcödé café", ["ünïcödé", "café"]),
        ("  multiple   spaces  ", ["multiple", "spaces"]),
    ],
)
def test_tokenize(text, expected):
    assert tokenize(text) == expected


# ---------- keep_alnum_and_spaces ----------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello, World! 123_ok", "Hello World 123ok"),
        ("a+b=c", "abc"),
        ("", ""),
        ("Ünïcödé café", "Ünïcödé café"),
    ],
)
def test_keep_alnum_and_spaces(text, expected):
    assert keep_alnum_and_spaces(text) == expected


# ---------- remove_stop_words ----------

@pytest.mark.parametrize(
    "text, stop_words, expected",
    [
        ("The Cat and THE dog!", ["the", "and"], "cat dog"),
        ("The Cat and THE dog!", ["The", "and"], "cat dog"),
        ("nothing to remove", [], "nothing to remove"),
        ("a b c", ["a", "b", "c"], ""),
    ],
)
def test_remove_stop_words(text, stop_words, expected):
    assert remove_stop_words(text, stop_words) == expected


# ---------- flatten ----------

@pytest.mark.parametrize(
    "list_of_lists, expected",
    [
        ([[1, 2], [3], []], [1, 2, 3]),
        ([[1, [2]], [3]], [1, [2], 3]),   # only one level is flattened
        ([], []),
        ([["a"], ["b", "c"]], ["a", "b", "c"]),
    ],
)
def test_flatten(list_of_lists, expected):
    assert flatten(list_of_lists) == expected


# ---------- shuffle_with_seed ----------

@pytest.mark.parametrize("seed", [0, 1, 42])
def test_shuffle_same_seed_same_order(numbers, seed):
    assert shuffle_with_seed(numbers, seed) == shuffle_with_seed(numbers, seed)


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_shuffle_is_permutation(numbers, seed):
    result = shuffle_with_seed(numbers, seed)
    assert sorted(result) == sorted(numbers)
    assert len(result) == len(numbers)


def test_shuffle_does_not_modify_input(numbers):
    original = list(numbers)
    shuffle_with_seed(numbers, 7)
    assert numbers == original


def test_shuffle_empty_list():
    assert shuffle_with_seed([], 3) == []