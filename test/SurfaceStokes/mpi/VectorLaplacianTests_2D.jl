using MPI, PartitionedArrays
using GridapGeosciences
include("../VectorLaplacian2D.jl")

MPI.Init()
nprocs = prod(MPI.Comm_size(MPI.COMM_WORLD))
ranks = distribute_with_mpi(LinearIndices((prod(nprocs),)))

n_ref_lvls = 4
radius = 1.0

## Distributed model: 2D
models = generate_distributed_refined_models(ranks, CubedSphereMesh(radius), n_ref_lvls, IntrinsicManifold())
VectorLaplacian2D.main(models;_i_am_main=i_am_main(ranks))

### P4test model: 2D
coarse_mesh = CubedSphereMesh(radius)
models = generate_octree_distributed_refined_models(ranks, coarse_mesh, n_ref_lvls, IntrinsicManifold())
VectorLaplacian2D.main(models;_i_am_main=i_am_main(ranks))
