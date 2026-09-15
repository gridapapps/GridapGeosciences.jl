using MPI, PartitionedArrays
using GridapGeosciences
include("../DarcyGMG.jl")


MPI.Init()
nprocs = prod(MPI.Comm_size(MPI.COMM_WORLD))
ranks = distribute_with_mpi(LinearIndices((prod(nprocs),)))

n_ref_lvls = 4
radius = 1.0

### P4test model: 2D -- Note needs to be omodel because we call ModelHierarchy
coarse_mesh = CubedSphereMesh(radius)
models = generate_octree_distributed_refined_models(ranks, coarse_mesh, n_ref_lvls, IntrinsicManifold())

# Block smoother:
DarcyGMG.main(models;smoother_type=:block,_i_am_main=i_am_main(ranks))

# Patch smoother:
DarcyGMG.main(models;smoother_type=:patch,_i_am_main=i_am_main(ranks))
