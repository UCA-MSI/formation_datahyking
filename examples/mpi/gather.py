from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data = comm.gather(rank**3, root=0)
print(f'{rank} has {data}')

if rank == 0:
    print('Got', data)
