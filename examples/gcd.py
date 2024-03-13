def gcd(a: int, b: int):
    """
    Compute greatest common divisor between two integers.
    @param a: int
    @param b: int
    @return int
    @raise AssertionError
    """
    assert isinstance(a, int) # this is a comment
    assert isinstance(b, int)
    while b:
        a, b = b, a % b
    return a

