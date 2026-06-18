using MPI, PartitionedArrays
using GridapGeosciences
include("../AmbientSurfaceArea.jl")

MPI.Init()
nprocs = prod(MPI.Comm_size(MPI.COMM_WORLD))
ranks = distribute_with_mpi(LinearIndices((prod(nprocs),)))

n_ref_lvls = 4
radius = 1.0

## Distributed model: 2D
parametric_models = get_distributed_intrinsic_cubed_sphere_refined_models(ranks,n_ref_lvls,radius)
ambient_models = get_distributed_extrinsic_cubed_sphere_refined_models(ranks,n_ref_lvls,radius)
AmbientSurfaceArea.main(parametric_models, ambient_models;_i_am_main=i_am_main(ranks))

# ### P4test model: 2D
# models = get_octree_ambient_refined_models(ranks,n_ref_lvls,radius)
# AmbientSurfaceArea.main(models;_i_am_main=i_am_main(ranks))