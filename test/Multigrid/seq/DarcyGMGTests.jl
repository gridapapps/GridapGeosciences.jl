include("../overload.jl")
include("../DarcyGMG.jl")

## Serial model: 2D
n_ref_lvls = 3
radius = 1.0
models = generate_refined_models(n_ref_lvls, CubedSphereMesh(radius), IntrinsicManifold())

# Block smoother:
main(models;smoother_type=:block)

# Patch smoother:
main(models;smoother_type=:patch)
