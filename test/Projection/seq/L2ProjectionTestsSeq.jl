include("../L2ProjectionTests.jl")

## Serial model: 2D
n_ref_lvls = 4
radius = 1.0
models = get_intrinsic_cubed_sphere_refined_models(n_ref_lvls,radius)
L2_projection(models)

## Serial model 3D
n_ref_lvls = 3
radius = 1.0
thickness = 0.19
coarse_mesh = CubedSphereWithThicknessMesh(radius,thickness)
models = get_refined_models(n_ref_lvls,coarse_mesh,IntrinsicManifold())
L2_projection(models)
