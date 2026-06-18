
using MPI, PartitionedArrays
using GridapGeosciences
include("../AmbientTransientWaveEquation.jl")

MPI.Init()
nprocs = prod(MPI.Comm_size(MPI.COMM_WORLD))
ranks = distribute_with_mpi(LinearIndices((prod(nprocs),)))

radius = 1.0
## Distributed model: 2D
models = get_distributed_extrinsic_cubed_sphere_refined_models(ranks,3,radius)
AmbientTransientWaveEquationTests.main(models[1];_i_am_main=i_am_main(ranks))

### P4test model: 2D
#omodel = CubedSphere2DAmbientOctreeDistributedDiscreteModel(ranks, radius;
#  num_initial_uniform_refinements=3)
#ambient_model = omodel.ambient_dmodel
#AmbientTransientWaveEquationTests.main(ambient_model;_i_am_main=i_am_main(ranks))
