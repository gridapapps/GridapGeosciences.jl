using MPI, PartitionedArrays
using GridapGeosciences
include("../AmbientLinearisedShallowWater.jl")

MPI.Init()
nprocs = prod(MPI.Comm_size(MPI.COMM_WORLD))
ranks = distribute_with_mpi(LinearIndices((prod(nprocs),)))

n_ref_lvls = 4
radius = 1.0

## Distributed model: 2D
models = get_distributed_extrinsic_cubed_sphere_refined_models(ranks,n_ref_lvls,radius)
AmbientLinearisedShallowWaterTests.main(models;_i_am_main=i_am_main(ranks))

# ### P4test model: 2D
#models = get_octree_ambient_refined_models(ranks,n_ref_lvls,radius)
#AmbientLinearisedShallowWaterTests.main(models;_i_am_main=i_am_main(ranks))

### P4test model: 3D
#thickness = 0.19
#models = get_3D_octree_ambient_refined_models(ranks,n_ref_lvls-1,radius,thickness)
#AmbientLinearisedShallowWaterTests.main(models;_i_am_main=i_am_main(ranks))
