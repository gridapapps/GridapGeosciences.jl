include("../VectorLaplacianSurface.jl")

## Serial model: 2D cubed sphere
n_ref_lvls = 4
radius = 1.0
models = generate_refined_models(n_ref_lvls, CubedSphereMesh(radius), IntrinsicManifold())
VectorLaplacianSurfaceTests.main(models; radius=radius)
