from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0:
    data = [{f'chunk_{i}': i for i in range(size)}]
    print('Original: ', data)
else:
    data = None

data = comm.bcast(data, root=0)
print(f'{rank} got: {data}')
