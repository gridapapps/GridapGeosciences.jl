"""
In this module, test the following skew operators in 2D geophysical applications:

Let ̃f and ̃u be a function and vector in the tangent space of the sphere
Let ̃k be the outward surface normal to the sphere

dagger:     ũ^†       = ̃k × ̃u           = J √g g^{-1} u^⟂
skew_grad:  ∇ᵧ^† ̃f    = ̃k × ∇ᵧ ̃f        = J grad^⟂ f / √g
skew_div:   ∇ᵧ^† ⋅ ̃u  = ∇ᵧ ⋅ ( ̃u × ̃k )  = -1/√g div( (√g(^2) g^{-1} u^⟂ )
"""

module SkewOperatorTests

using GridapGeosciences
using Gridap
using Test

# Model definition
radius = 1
n_ref_lvls = 1
ambient_model = CubedSphereAmbientDiscreteModel(radius;num_initial_uniform_refinements=n_ref_lvls)
panel_model = get_parametric_model(ambient_model)


## Test we obtain the surface normal for the ambient model
Ω_ambient = Triangulation(ambient_model)
pts_ambient = get_cell_points(Ω_ambient)
n_ambient = get_surface_normal(Ω_ambient)
@test true

## Test the surface normal for the parametric model breaks
Ω_panel = Triangulation(panel_model)
pts_panel = get_cell_points(Ω_panel)
@test_skip (@test_broken get_surface_normal(Ω_panel) )

## Function
function ambient_f(x)
  x[1]*x[2]*x[3]
end
panel_f = panel_to_cartesian(ambient_f)

## Vector in the tangent space of the sphere
function ambient_vec(x)
  VectorValue(-x[2],x[1],0)
end
panel_vec = panel_to_cartesian(ambient_vec)


################################################################################
########## Dagger operator
################################################################################

### Compute u^† = k × u for the ambient model
u_cf_ambient = CellField(ambient_vec,Ω_ambient)
u_dagger_ambient = dagger(u_cf_ambient)

### Compute u^† = J *(R g u / √g)  for the parametric model
u_cf_panel = ParametricCellField(contra_v(panel_vec),Ω_panel)
J_cf = ParametricCellField(forward_jacobian,Ω_panel)
meas_cf = ParametricCellField(sqrtg,Ω_panel)
metric_cf = ParametricCellField(metric,Ω_panel)
inv_metric_cf = ParametricCellField(inv_metric,Ω_panel)
u_dagger_panel = J_cf ⋅ (inv_metric_cf ⋅ perp(u_cf_panel))*(meas_cf)

### Test u^† for the parametric and ambient model are equivalent
@test all(u_dagger_panel(pts_panel) .≈ u_dagger_ambient(pts_ambient))


################################################################################
########## Skew gradient
################################################################################
sgrad_cf_ambient = AmbientCellField(ambient_sgrad(ambient_f),Ω_ambient)
skew_grad_ambient = n_ambient × sgrad_cf_ambient

f_cf_panel = ParametricCellField(panel_f,Ω_panel)
skew_grad_panel = J_cf ⋅( perp(gradient(f_cf_panel))   )*(1.0/meas_cf)

### Test the maximum cellwise difference of ∇ᵧ^† f is machine eps
dif = skew_grad_ambient(pts_ambient) .- skew_grad_panel(pts_panel)
max_dif = map(x->maximum(norm.(x)),dif)
@test all(max_dif .< 1e-12)


################################################################################
########## Skew divergence
################################################################################
detg_cf = ParametricCellField(detg,Ω_panel)
_in = detg_cf*(inv_metric_cf⋅perp(u_cf_panel))
skew_div_panel = -(1.0/meas_cf)*divergence(_in)

vcrossk(x) = ambient_vec(x) × normal_vec(x)
skew_div_ambient = AmbientCellField(ambient_surfdiv(vcrossk),Ω_ambient)

### Test ∇ᵧ^† ⋅ u is equivalent for ambient and panel
@test all(skew_div_ambient(pts_ambient) .≈ skew_div_panel(pts_panel))





end # module
