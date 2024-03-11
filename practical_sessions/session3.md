# Unit testing

## Exercise
You're give the following simple function:

```
# prime.py
from math import sqrt, floor

def is_prime(n):
    if n <= 3:
        return n > 1
    if n % 2 == 0 or n % 3 == 0:
        return False
    limit = floor(sqrt(n))
    for i in range(5, limit+1, 6):
        if n % i == 0 or n % (i+2) == 0:
            return False
    return True

```

Create a new file: `test_prime.py`:

```
import unittest
from prime import is_prime

class TestPrime(unittest.TestCase):
    def test_is_prime(self):
        ...
```

### Part 1:

Write test(s) to check:

  * correctness of the algorithm

### Part 2:

Extend the functionalities of the `is_prime` function with the following features.

   1. When the attribute passed is a string it must raise a `TypeError`
   2. When the attribute passed is a float is must raise a `ValueError`
   3. When the attribute passed is a negative integer it must raise a `ValueError`
   4. The two `ValueError` must be differentiated (different text)
   
Complete the test suite to check the correctness of your implementation.