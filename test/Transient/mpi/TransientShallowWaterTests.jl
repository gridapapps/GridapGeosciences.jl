
using MPI, PartitionedArrays
using GridapGeosciences
include("../TransientShallowWater.jl")

MPI.Init()
nprocs = prod(MPI.Comm_size(MPI.COMM_WORLD))
ranks = distribute_with_mpi(LinearIndices((prod(nprocs),)))

radius = 1.0
## Distributed model: 2D
models = generate_distributed_refined_models(ranks, CubedSphereMesh(radius), 3, IntrinsicManifold())
TransientShallowWaterTests.main(models[1];_i_am_main=i_am_main(ranks))

### P4test model: 2D
# omodel = CubedSphere2DParametricOctreeDistributedDiscreteModel(ranks, radius;
#  num_initial_uniform_refinements=3)
#panel_model = omodel.parametric_dmodel
#TransientShallowWaterTests.main(panel_model;_i_am_main=i_am_main(ranks))
