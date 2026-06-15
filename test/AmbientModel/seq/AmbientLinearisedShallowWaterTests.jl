include("../AmbientLinearisedShallowWater.jl")

## Serial model: 2D
n_ref_lvls = 4
radius = 1.0
extrinsic_models = get_extrinsic_cubed_sphere_refined_models(n_ref_lvls,radius)
AmbientLinearisedShallowWaterTests.main(extrinsic_models)

# ### I do not like having this here, but need to think of a better way
# to compare one error result to the intrinsic approach

extrinsic_model = extrinsic_models[1]
dir = @__DIR__
p_fe = 1

h = AmbientLinearisedShallowWaterTests.h₀(0.0)
vX = tangent_vec(AmbientLinearisedShallowWaterTests.u₀(0.0))
f = AmbientLinearisedShallowWaterTests.f₀(0.0)
e_u_ambient, e_p_ambient, = AmbientLinearisedShallowWaterTests.linear_shallow_water_solver(
  extrinsic_model,p_fe,dir,
  h,vX,f)


include("../../Geophysical/LinearisedShallowWater.jl")

intrinsic_models = get_intrinsic_cubed_sphere_refined_models(n_ref_lvls,radius)
intrinsic_model = intrinsic_models[1]
e_u_panel, e_p_panel, = LinearisedShallowWaterTests.linear_shallow_water_solver(
  intrinsic_model,p_fe,dir,
  h,vX,f)

e_u_comparison = e_u_ambient - e_u_panel
println(e_u_comparison)
@test e_u_comparison < 1e-12

e_p_comparison = e_p_ambient - e_p_panel
println(e_p_comparison)
@test e_p_comparison < 1e-12
