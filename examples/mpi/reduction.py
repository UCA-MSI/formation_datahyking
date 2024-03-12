from mpi4py import MPI 

def reduction(a,b):
    return a + b

comm = MPI.COMM_WORLD
worker = comm.Get_rank()
size = comm.Get_size()

data = comm.reduce(worker, op=reduction, root=0)
print(f"worker {worker} has {worker}")

if worker == 0:
    print("Final result: ", data)