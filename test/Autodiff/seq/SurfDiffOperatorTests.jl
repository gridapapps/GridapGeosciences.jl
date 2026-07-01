"""
In this module, test that ∇s, divs, and Δs produce the same result for
ExtrinsicManifold (ambient) and IntrinsicManifold (parametric) AtlasDiscreteModels.
"""

module SurfDiffOperatorTests

using GridapGeosciences
using Gridap
using Test

radius    = 1.0
n_ref_lvls = 1
coarse_mesh = CubedSphereMesh(radius)

ambient_model    = AtlasDiscreteModel(coarse_mesh, n_ref_lvls; manifold_style=ExtrinsicManifold())
parametric_model = AtlasDiscreteModel(coarse_mesh, n_ref_lvls; manifold_style=IntrinsicManifold())

Ω_ambient    = Triangulation(ambient_model)
Ω_parametric = Triangulation(parametric_model)
pts_ambient    = get_cell_points(Ω_ambient)
pts_parametric = get_cell_points(Ω_parametric)

function ambient_fX(xyz)
  x, y, z = xyz
  x^2 + y^2*z
end

function ambient_vecX(xyz)
  x, y, _ = xyz
  VectorValue(y^2, -x^2, 0.0)
end

################################################################################
########## Surface gradient: ∇s
################################################################################

sgrad_ambient    = ∇s(ambient_fX, Ω_ambient)
sgrad_parametric = ∇s(ambient_fX, Ω_parametric)
dif = sgrad_ambient(pts_ambient) .- sgrad_parametric(pts_parametric)
max_dif = map(x -> maximum(norm.(x)), dif)
@test all(max_dif .< 1e-12)

################################################################################
########## Surface divergence: divs
################################################################################

sdiv_ambient    = divs(ambient_vecX, Ω_ambient)
sdiv_parametric = divs(ambient_vecX, Ω_parametric)
dif = sdiv_ambient(pts_ambient) .- sdiv_parametric(pts_parametric)
max_dif = map(x -> maximum(abs.(x)), dif)
@test all(max_dif .< 1e-12)

################################################################################
########## Surface Laplacian: Δs
################################################################################

slap_ambient    = Δs(ambient_fX, Ω_ambient)
slap_parametric = Δs(ambient_fX, Ω_parametric)
dif = slap_ambient(pts_ambient) .- slap_parametric(pts_parametric)
max_dif = map(x -> maximum(abs.(x)), dif)
@test all(max_dif .< 1e-12)

################################################################################
########## AD vs no-AD for the parametric model
################################################################################

slap_parametric_ad = Δs(ambient_fX, Ω_parametric; use_automatic_differentiation=true)
dif = slap_parametric(pts_parametric) .- slap_parametric_ad(pts_parametric)
max_dif = map(x -> maximum(abs.(x)), dif)
@test all(max_dif .< 1e-12)

end # module
