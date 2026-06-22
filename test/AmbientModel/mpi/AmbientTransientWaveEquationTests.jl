
using MPI, PartitionedArrays
using GridapGeosciences
include("../AmbientTransientWaveEquation.jl")

MPI.Init()
nprocs = prod(MPI.Comm_size(MPI.COMM_WORLD))
ranks = distribute_with_mpi(LinearIndices((prod(nprocs),)))

radius = 1.0
## Distributed model: 2D
models = generate_distributed_refined_models(ranks, CubedSphereMesh(radius), 3, ExtrinsicManifold())
AmbientTransientWaveEquationTests.main(models[1];_i_am_main=i_am_main(ranks))

### P4test model: 2D
omodel = AtlasOctreeDistributedDiscreteModel(ranks, CubedSphereMesh(radius), 3; manifold_style=ExtrinsicManifold())
ambient_model = get_atlas_model(omodel)
AmbientTransientWaveEquationTests.main(ambient_model;_i_am_main=i_am_main(ranks))

