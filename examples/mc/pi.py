# estimate pi with montecarlo simulation
import random
import matplotlib.pyplot as plt
import numpy as np
import time

def estimate_pi(num_samples):
    cnt = 0
    for i in range(num_samples):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            cnt += 1
    pi_estimate = 4 * cnt / num_samples
    return pi_estimate

num_samples = 10000
# Estimate the value of pi using 10,000 samples
pi_estimate = estimate_pi(num_samples)
print("Estimated value of pi:", pi_estimate)

x = np.random.uniform(-1, 1, num_samples)
y = np.random.uniform(-1, 1, num_samples)
plt.plot(x, y, 'k.', markersize=1)

t = np.linspace(0, np.pi/2, 100) # a quarter circle
xc = np.cos(t)
yc = np.sin(t)
plt.plot(xc, yc, 'r-', linewidth=2)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('x')
plt.ylabel('y')

plt.title(f'{pi_estimate:.4f}')
plt.show()

