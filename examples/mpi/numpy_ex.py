import numpy as np


if __name__ == "__main__":

    n = 10000000

    a = np.ones(n)
    b = np.zeros(n)
    for i in range(n):
        b[i] = 1.0 + i 

    for i in range(n):
        a[i] = a[i] + b[i]
    sum_ = 0.0
    for i in range(n):
        sum_ += a[i]

    average = sum_ / n
    print(f'Average: {average}')
