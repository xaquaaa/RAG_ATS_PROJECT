from scripts.analyze_confidence_scores import percentile


def test_percentile_basic():
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert percentile(values, 0.0) == 0.1
    assert percentile(values, 1.0) == 0.5


def test_percentile_median_ish():
    values = [1.0, 2.0, 3.0, 4.0]
    # index = int(4 * 0.5) = 2 -> values[2] = 3.0
    assert percentile(values, 0.5) == 3.0


def test_percentile_unsorted_input_is_sorted_internally():
    values = [0.5, 0.1, 0.3]
    assert percentile(values, 0.0) == 0.1


def test_percentile_empty_list():
    import math
    assert math.isnan(percentile([], 0.5))
