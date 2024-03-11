# Concurrency and parallelisation

## Live coding
### Part 1a:

  * Concurrency
    * file: `examples/echo_server.py`

### Part 1b: performance via data structures

  * Performance Tests
    * file: `examples/queuing.py`
  
### Part 2:

  * Montecarlo simulation
    * file: `example/mc/pi.py`

We will estimate the value of $\pi$ by using the Montecarlo method.

*Approach*: We start from a circle `C` of radius `r = 1` inscribed in a square `S` of size `l = 2`.
Then we generate a number of random points and check how many fall in the enclosed circle.

$$
Area(S) = 4 \\
Area(C) = \pi 
$$

If we calculate the ratio between the number of points that falls into the circle and the total
number of samples we get an approximation of $\pi$.

$$
\frac{C}{S} = \frac{\pi}{4} \\
$$

Therefore:

$$
\pi = 4 * \frac{C}{S}
$$

  
### Part 3:

  * Parallelisation
    * file: `example/mpi/numpy_ex.py`
  
Increase performances via MPI.