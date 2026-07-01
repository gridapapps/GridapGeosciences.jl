"""
In this module, test the skeleton normal vectors for the ambient model are
  1. in the tangent space of the sphere
  2. equiv to the mapped skeleton normal vectors of the parametric model
"""

module AmbientNormalVectorTests

using Gridap
using GridapGeosciences
using Test

radius = 1.0
n_ref_lvls = 1
coarse_mesh = CubedSphereMesh(radius)
ambient_model = AtlasDiscreteModel(coarse_mesh,n_ref_lvls; manifold_style=ExtrinsicManifold())
parametric_model = AtlasDiscreteModel(coarse_mesh,n_ref_lvls; manifold_style=IntrinsicManifold())

################################################################################
########## Ambient model
################################################################################
Ω_ambient = Triangulation(ambient_model)
n_surface_ambient = get_sphere_surface_normal(Ω_ambient)

Λ_ambient = SkeletonTriangulation(ambient_model)
n_Λ_ambient = get_normal_vector(Λ_ambient)
pts_ambient = get_cell_points(Λ_ambient)

### Test the skeleton normal is in the tangent space of the sphere
normal_component = (n_Λ_ambient⋅n_surface_ambient).plus(pts_ambient)
max_dif = map(x->maximum(norm.(x)),normal_component)
@test all(max_dif .< 1e-12)

normal_component = (n_Λ_ambient⋅n_surface_ambient).minus(pts_ambient)
max_dif = map(x->maximum(norm.(x)),normal_component)
@test all(max_dif .< 1e-12)

################################################################################
########## Parametric model
################################################################################
Λ_panel = SkeletonTriangulation(parametric_model)
n_Λ_mapped = pushforward_normal(Λ_panel)
pts_panel = get_cell_points(Λ_panel)

## Test equivalence with ambient model
dif = n_Λ_mapped.plus(pts_panel) - n_Λ_ambient.plus(pts_ambient)
max_dif = map(x->maximum(norm.(x)),dif)
@test all(max_dif .< 1e-12)

dif = n_Λ_mapped.minus(pts_panel) - n_Λ_ambient.minus(pts_ambient)
max_dif = map(x->maximum(norm.(x)),dif)
@test all(max_dif .< 1e-12)


@test true

end # module
