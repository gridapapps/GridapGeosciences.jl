using MPI, PartitionedArrays
using GridapGeosciences
include("../Hierarchy.jl")

MPI.Init()
nprocs = prod(MPI.Comm_size(MPI.COMM_WORLD))
ranks = distribute_with_mpi(LinearIndices((prod(nprocs),)))

n_ref_lvls = 3
radius = 1.0
### P4test model: 2D
coarse_mesh = CubedSphereMesh(radius)
coarse_model = AtlasOctreeDistributedDiscreteModel(ranks, coarse_mesh, 0; manifold_style=IntrinsicManifold())
HierarchyTest.main(coarse_model, n_ref_lvls)
