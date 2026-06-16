
using MPI, PartitionedArrays
using GridapGeosciences
include("../TransientWaveEquation.jl")

MPI.Init()
nprocs = prod(MPI.Comm_size(MPI.COMM_WORLD))
ranks = distribute_with_mpi(LinearIndices((prod(nprocs),)))

radius = 1.0
## Distributed model: 2D
models = get_distributed_intrinsic_cubed_sphere_refined_models(ranks,3,radius)
TransientWaveEquationTests.main(models[1];_i_am_main=i_am_main(ranks))

### P4test model: 2D
#omodel = CubedSphere2DParametricOctreeDistributedDiscreteModel(ranks, radius;
#  num_initial_uniform_refinements=3)
#panel_model = omodel.parametric_dmodel
# TransientWaveEquationTests.main(panel_model;_i_am_main=i_am_main(ranks))
