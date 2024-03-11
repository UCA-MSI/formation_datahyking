from collections import deque
from time import perf_counter

TIMES = 100_000
lst = []
deq = deque()

def avg_time(func, n):
    """
    Apply `func` for `n` times, measuring the time perf.
    @param func: function
    @param n: int
    @return float: average time per function call
    """
    total = 0.0
    for i in range(n):
        start = perf_counter()
        func(i)
        total += (perf_counter() - start)
    return total / n

lst_time = avg_time(lambda i: lst.insert(0, i), TIMES)
deq_time = avg_time(lambda i: deq.appendleft(i), TIMES)
speedup = lst_time / deq_time

print(f"list.insert()      {lst_time:.6} ns")
print(f"deque.appendleft() {deq_time:.6} ns  (x{speedup:.6})")


# Exercise: modify the above code to evaluate the performances of lst.pop(0) versus deq.popleft() 