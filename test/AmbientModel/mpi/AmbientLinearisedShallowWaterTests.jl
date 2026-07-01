using MPI, PartitionedArrays
using GridapGeosciences
include("../AmbientLinearisedShallowWater.jl")

MPI.Init()
nprocs = prod(MPI.Comm_size(MPI.COMM_WORLD))
ranks = distribute_with_mpi(LinearIndices((prod(nprocs),)))

n_ref_lvls = 4
radius = 1.0

## Distributed model: 2D
models = generate_distributed_refined_models(ranks, CubedSphereMesh(radius), n_ref_lvls, ExtrinsicManifold())
AmbientLinearisedShallowWaterTests.main(models;_i_am_main=i_am_main(ranks))

### P4test model: 2D
coarse_mesh = CubedSphereMesh(radius)
models = generate_octree_distributed_refined_models(ranks, coarse_mesh, n_ref_lvls, ExtrinsicManifold())
AmbientLinearisedShallowWaterTests.main(models;_i_am_main=i_am_main(ranks))

### P4test model: 3D
thickness = 0.19
coarse_mesh = CubedSphereWithThicknessMesh(radius, thickness)
models = generate_octree_distributed_refined_models(ranks, coarse_mesh, n_ref_lvls-2, ExtrinsicManifold())
AmbientLinearisedShallowWaterTests.main(models;_i_am_main=i_am_main(ranks))

### P4test model: Extruded 3D
coarse_mesh = ExtrudedCubedSphereWithThicknessMesh(radius, thickness)
models = generate_extruded_octree_distributed_refined_models(ranks, coarse_mesh, n_ref_lvls-2, ExtrinsicManifold())
AmbientLinearisedShallowWaterTests.main(models;_i_am_main=i_am_main(ranks))
