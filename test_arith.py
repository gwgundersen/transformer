import pytest

from transformer.arith import carry_count

# --------------------------------------------------------------------------------------------------

# Each case is (x, y, expected).
CARRY_COUNT_CASES = [
    (1, 1, 0),
    (1, 8, 0),
    (1, 9, 1),
    (1, 19, 1),
    (145, 255, 2),
    (545, 455, 3),
]


@pytest.mark.parametrize("x, y, expected", CARRY_COUNT_CASES)
def test_carry_count(x: int, y: int, expected: int) -> None:
    assert carry_count(x, y) == expected
