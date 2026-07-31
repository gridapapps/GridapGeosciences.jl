include("../VectorLaplacian2D.jl")

## Serial model: 2D
n_ref_lvls = 4
radius = 1.0
models = generate_refined_models(n_ref_lvls, CubedSphereMesh(radius), IntrinsicManifold())
main(models;ps=[2])
# VectorLaplacian2D.main(models)
