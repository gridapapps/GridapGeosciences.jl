using MPI, PartitionedArrays
using GridapGeosciences
include("../HodgeLaplacian_vector.jl")

MPI.Init()
nprocs = prod(MPI.Comm_size(MPI.COMM_WORLD))
ranks = distribute_with_mpi(LinearIndices((prod(nprocs),)))

n_ref_lvls = 3
radius = 1.0
thickness = 0.19
coarse_mesh = CubedSphereWithThicknessMesh(radius, thickness)
models = generate_distributed_refined_models(ranks, coarse_mesh, n_ref_lvls, IntrinsicManifold())
HodgeLaplacianVectorTests.main(models;_i_am_main=i_am_main(ranks))

## P4test model: 3D
models = generate_octree_distributed_refined_models(ranks, coarse_mesh, n_ref_lvls-1, IntrinsicManifold())
HodgeLaplacianVectorTests.main(models;_i_am_main=i_am_main(ranks))
