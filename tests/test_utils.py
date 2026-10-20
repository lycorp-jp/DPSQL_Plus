import pytest

from dpsql.errors import AggregationError
from dpsql.utils import safely_get_threshold, validate_clipping_bounds


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


@pytest.mark.parametrize(
    "bounds",
    [
        (-float("inf"), 1.0),
        (0.0, float("inf")),
        (0.0, float("nan")),
        (0.0, 0.0),
        (2.0, 1.0),
    ],
)
def test_validate_clipping_bounds_rejects_invalid_bounds(bounds):
    with pytest.raises(AggregationError):
        validate_clipping_bounds(bounds)
