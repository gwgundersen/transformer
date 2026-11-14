import pytest

from transformer.arith import count_carry_cascades

# --------------------------------------------------------------------------------------------------

# Each case is (x, y, expected).
CARRY_COUNT_CASES = [
    # no carries
    (1, 1, 0),
    (1, 8, 0),
    (9, 990, 0),
    # one carry but no cascade
    (1, 9, 0),
    (10, 90, 0),
    (100, 900, 0),
    (55, 54, 0),
    (113, 117, 0),
    (131, 171, 0),
    (311, 711, 0),
    # one carry, one cascade
    (45, 55, 1),
    (145, 155, 1),
    # two carries, zero cascades
    (101, 909, 0),
    (501, 589, 0),
    # two carries, one cascade
    (501, 599, 1),
    # two carries, two cascades
    (401, 599, 2),
    # three carries, zero cascades
    (555, 555, 0),
    # three carries, one cascade
    (595, 505, 1),
    # three carries, two cascades
    (445, 555, 2),
]


@pytest.mark.parametrize("x, y, expected", CARRY_COUNT_CASES)
def test_count_carry_cascades(x: int, y: int, expected: int) -> None:
    assert count_carry_cascades(x, y) == expected
