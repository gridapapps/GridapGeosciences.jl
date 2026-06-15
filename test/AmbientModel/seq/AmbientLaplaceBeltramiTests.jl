include("../AmbientLaplaceBeltrami.jl")

## Serial model: 2D
n_ref_lvls = 4
radius = 1.0
extrinsic_models = get_extrinsic_cubed_sphere_refined_models(n_ref_lvls,radius)
AmbientLaplaceBeltrami.main(extrinsic_models)

# ### I do not like having this here, but need to think of a better way
# to compare one error result to the intrinsic approach

extrinsic_model = extrinsic_models[1]
dir = @__DIR__
p_fe = 2
e_ambient, = AmbientLaplaceBeltrami.laplace_beltrami_solver(
              extrinsic_model,p_fe,dir,
              AmbientLaplaceBeltrami.fX)


include("../../Laplacian/LaplaceBeltrami.jl")

intrinsic_models = get_intrinsic_cubed_sphere_refined_models(n_ref_lvls,radius)
intrinsic_model = intrinsic_models[1]
e_panel, = LaplaceBeltramiTests.laplace_beltrami_solver(
              intrinsic_model,p_fe,dir,
              AmbientLaplaceBeltrami.fX)

e_comparison = e_ambient - e_panel
println(e_comparison)
@test e_comparison < 1e-12
