using MPI, PartitionedArrays
using GridapGeosciences
include("../HodgeLaplacian_scalar.jl")

MPI.Init()
nprocs = prod(MPI.Comm_size(MPI.COMM_WORLD))
ranks = distribute_with_mpi(LinearIndices((prod(nprocs),)))

n_ref_lvls = 4
radius = 1.0

## Distributed model: 2D
models = get_distributed_intrinsic_cubed_sphere_refined_models(ranks,n_ref_lvls,radius)
HodgeLaplacianScalarTests.main(models;_i_am_main=i_am_main(ranks))

## Distributed model: 3D
n_ref_lvls = 3
radius = 1.0
thickness = 0.19
coarse_mesh = CubedSphereWithThicknessMesh(radius, thickness)
models = get_distributed_refined_models(ranks, coarse_mesh, n_ref_lvls, IntrinsicManifold())
HodgeLaplacianScalarTests.main(models;_i_am_main=i_am_main(ranks))

### P4test model: 2D
# models = get_octree_refined_models(ranks,n_ref_lvls,radius)
# HodgeLaplacianScalarTests.main(models;_i_am_main=i_am_main(ranks))

### P4test model: 3D
# thickness = 0.19
# models = get_3D_octree_refined_models(ranks,n_ref_lvls-1,radius,thickness)
# HodgeLaplacianScalarTests.main(models;ps=[1],_i_am_main=i_am_main(ranks))
