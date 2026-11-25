include("../DarcyGMG.jl")

## Serial model: 2D
n_ref_lvls = 4
radius = 1.0
models = generate_refined_models(n_ref_lvls, CubedSphereMesh(radius), IntrinsicManifold())

# Block smoother:
DarcyGMG.main(models;smoother_type=:block)

# Patch smoother:
DarcyGMG.main(models;smoother_type=:patch)
