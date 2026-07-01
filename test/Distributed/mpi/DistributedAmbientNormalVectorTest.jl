using MPI, PartitionedArrays
include("../DistributedAmbientNormalVectorTests.jl")

MPI.Init()
nprocs = prod(MPI.Comm_size(MPI.COMM_WORLD))

with_mpi() do distribute
  DistributedAmbientNormalTests.main(distribute,nprocs)
end
