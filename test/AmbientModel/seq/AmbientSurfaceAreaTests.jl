include("../AmbientSurfaceArea.jl")

## Serial model: 2D
n_ref_lvls = 4
radii = [1.0, 2.0]

for radius in radii
  ambient_models = get_extrinsic_cubed_sphere_refined_models(n_ref_lvls,radius)
  parametric_models = get_intrinsic_cubed_sphere_refined_models(n_ref_lvls,radius)
  AmbientSurfaceArea.main(parametric_models, ambient_models)
end
