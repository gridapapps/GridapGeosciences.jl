"""
In this module, test the following skew operators in 2D geophysical applications:

Let ̃f and ̃u be a function and vector in the tangent space of the sphere
Let ̃k be the outward surface normal to the sphere

dagger:     ũ^†       = ̃k × ̃u           = J √g g^{-1} u^⟂
skew_grad:  ∇ᵧ^† ̃f    = ̃k × ∇ᵧ ̃f        = J grad^⟂ f / √g
skew_div:   ∇ᵧ^† ⋅ ̃u  = ∇ᵧ ⋅ ( ̃u × ̃k )  = -1/√g div( (√g(^2) g^{-1} u^⟂ )
"""

module SkewOperatorTests

using GridapGeosciences
using Gridap
using Test

# Model definition
radius = 1.0
n_ref_lvls = 1
coarse_mesh = CubedSphereMesh(radius)
ambient_model = AtlasDiscreteModel(coarse_mesh,n_ref_lvls; manifold_style=ExtrinsicManifold())
parametric_model = AtlasDiscreteModel(coarse_mesh,n_ref_lvls; manifold_style=IntrinsicManifold())


## Test we obtain the surface normal for the ambient model
Ω_ambient = Triangulation(ambient_model)
pts_ambient = get_cell_points(Ω_ambient)
n_ambient = get_surface_normal(Ω_ambient)
@test true

## Test the surface normal for the parametric model breaks
Ω_parametric = Triangulation(parametric_model)
pts_parametric = get_cell_points(Ω_parametric)
@test_skip (@test_broken get_surface_normal(Ω_parametric) )

## Function
function ambient_f(x)
  x[1]*x[2]*x[3]
end

## Vector in the tangent space of the sphere
function ambient_vec(x)
  VectorValue(-x[2],x[1],0)
end

## Atlas cell fields for the parametric (intrinsic) model
ambient_map_cf = AmbientMapCellField(Ω_parametric)
J_cf           = transpose∘∇(ambient_map_cf)               # J^T : chart → ambient covariant basis
meas_cf        = MeasureCellField(Ω_parametric)            # √det g
inv_metric_cf  = InvMetricCellField(Ω_parametric)          # g^{-1}

################################################################################
########## Dagger operator
################################################################################

### Compute u^† = k × u for the ambient model
u_cf_ambient = CellField(ambient_vec,Ω_ambient)
u_dagger_ambient = dagger(u_cf_ambient)

### Compute u^† = J g^{-1} u^⟂ √g  for the parametric model
u_dagger_parametric = dagger(ambient_vec, Ω_parametric)
u_dagger_parametric(pts_parametric)

### Test u^† for the parametric and ambient model are equivalent
@test all(u_dagger_parametric(pts_parametric) .≈ u_dagger_ambient(pts_ambient))


################################################################################
########## Skew gradient
################################################################################
sgrad_cf_ambient = ∇s(ambient_f, Ω_ambient)
skew_grad_ambient = n_ambient × sgrad_cf_ambient
skew_grad_parametric = skew_∇s(ambient_f, Ω_parametric)

### Test the maximum cellwise difference of ∇ᵧ^† f is machine eps
dif = skew_grad_ambient(pts_ambient) .- skew_grad_parametric(pts_parametric)
max_dif = map(x->maximum(norm.(x)),dif)
@test all(max_dif .< 1e-12)


################################################################################
########## Skew divergence
################################################################################
# By now, automatic_differentiation=false does NOT work for the skew divergence,
# so we only test the AD version here
skew_div_parametric = skew_divs(ambient_vec, Ω_parametric; use_automatic_differentiation=true)
vcrossk(x) = ambient_vec(x) × sphere_surface_normal_vec(x)
skew_div_ambient = divs(vcrossk, Ω_ambient)

### Test ∇ᵧ^† ⋅ u is equivalent for ambient and parametric
@test all(skew_div_ambient(pts_ambient) .≈ skew_div_parametric(pts_parametric))

end # module
