include("../Hierarchy.jl")

## Serial model: 2D
n_ref_lvls = 3
radius = 1.0
coarse_mesh = CubedSphereMesh(radius)
coarse_model = AtlasDiscreteModel(coarse_mesh,0,manifold_style=IntrinsicManifold())
HierarchyTest.main(coarse_model,n_ref_lvls)
