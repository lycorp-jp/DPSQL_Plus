import pytest

from dpsql.errors import AggregationError
from dpsql.utils import safely_get_threshold


def test_safely_get_threshold():
    clipping_threshold = [(-10.0, 10.0), (0.0, 5.0)]

    # Test valid index
    lower, upper = safely_get_threshold(clipping_threshold, 0, "SUM")
    assert lower == -10.0
    assert upper == 10.0

    lower, upper = safely_get_threshold(clipping_threshold, 1, "SUM")
    assert lower == 0.0
    assert upper == 5.0

    # Test invalid index
    with pytest.raises(AggregationError):
        safely_get_threshold(clipping_threshold, 2, "SUM")
    with pytest.raises(AggregationError):
        safely_get_threshold(None, 0, "SUM")
