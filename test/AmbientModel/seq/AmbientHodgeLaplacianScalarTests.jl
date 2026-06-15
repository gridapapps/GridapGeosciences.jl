include("../AmbientHodgeLaplacian_scalar.jl")

## Serial model: 2D
n_ref_lvls = 4
radius = 1.0
models = get_extrinsic_cubed_sphere_refined_models(n_ref_lvls,radius)
AmbientHodgeLaplacianScalarTests.main(models)



# ### I do not like having this here, but need to think of a better way
# to compare one error result to the intrinsic approach
ambient_model = models[1]
dir = @__DIR__
p_fe = 2
e_ambient, = AmbientHodgeLaplacianScalarTests.hodge_laplacian_scalar(
              ambient_model,p_fe,dir,
              AmbientHodgeLaplacianScalarTests.fX)


include("../../Laplacian/HodgeLaplacian_scalar.jl")

intrinsic_models = get_intrinsic_cubed_sphere_refined_models(n_ref_lvls,radius)
intrinsic_model = intrinsic_models[1]
e_panel, = HodgeLaplacianScalarTests.hodge_laplacian_scalar(
              intrinsic_model,p_fe,dir,
              AmbientHodgeLaplacianScalarTests.fX)

e_comparison = e_ambient - e_panel
println(e_comparison)
@test e_comparison < 1e-12
