using MPI, PartitionedArrays
using GridapGeosciences
include("../LinearBoussinesq.jl")

MPI.Init()
nprocs = prod(MPI.Comm_size(MPI.COMM_WORLD))
ranks = distribute_with_mpi(LinearIndices((prod(nprocs),)))

n_ref_lvls = 3

### P4test model: 3D
# radius = 1.0
# thickness = 0.19
# models = get_3D_octree_refined_models(ranks,n_ref_lvls,radius,thickness)
# LinearisedBoussinesqTests.main(models;_i_am_main=i_am_main(ranks))
