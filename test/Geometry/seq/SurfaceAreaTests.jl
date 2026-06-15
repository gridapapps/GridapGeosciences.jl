include("../SurfaceArea.jl")

## Serial model: 2D
n_ref_lvls = 4
radii = [1.0, 2.0]

for radius in radii
  parametric_models = get_intrinsic_cubed_sphere_refined_models(n_ref_lvls,radius)
  SurfaceArea.main(parametric_models)
end
