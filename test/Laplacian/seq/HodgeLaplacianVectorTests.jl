include("../HodgeLaplacian_vector.jl")

## Serial model: 3D
n_ref_lvls = 3
radius = 1.0
thickness = 0.19
coarse_mesh = CubedSphereWithThicknessMesh(radius,thickness)
models = generate_refined_models(n_ref_lvls,coarse_mesh,IntrinsicManifold())
HodgeLaplacianVectorTests.main(models)
