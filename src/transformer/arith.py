def count_carry_cascades(x: int, y: int) -> int:
    """
    A "cascade" is a carry induced by an incoming carry digit. For example, 999+999 has three
    carries, but each one is locally knowable. You don't have to look ahead. Contrast that with
    445+555. Here, 4+5 and 5+4 do not carry. You must see 5+5 before you see the cascade. These
    problems are trickier for transformers.
    """
    count = 0
    carry = 0
    xz = str(x).zfill(3)
    yz = str(y).zfill(3)
    # Walk digits right-to-left
    for xd, yd in zip(reversed(xz), reversed(yz)):
        ad = int(xd) + int(yd)
        if ad == 9 and carry:
            count += 1
        carry = int((ad + carry) >= 10)
    return count
