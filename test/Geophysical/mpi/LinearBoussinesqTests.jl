using MPI, PartitionedArrays
using GridapGeosciences
include("../LinearBoussinesq.jl")

MPI.Init()
nprocs = prod(MPI.Comm_size(MPI.COMM_WORLD))
ranks = distribute_with_mpi(LinearIndices((prod(nprocs),)))

n_ref_lvls = 4
radius = 1.0
thickness = 0.19

### P4test model: Extruded 3D
coarse_mesh = ExtrudedCubedSphereWithThicknessMesh(radius, thickness)
models = generate_extruded_octree_distributed_refined_models(ranks, coarse_mesh, n_ref_lvls-2, IntrinsicManifold())
LinearisedBoussinesqTests.main(models;_i_am_main=i_am_main(ranks))
