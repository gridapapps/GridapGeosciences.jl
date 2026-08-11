"""
This test is related to https://github.com/gridapapps/GridapGeosciences.jl/issues/59

In this test, want to ensure the following two approaches are equivalent:
1. compute the ambient solution and then restrict to the plus and minus sides in the ambient space.
2. restrict to the plus and minus side in the chart, and push with a Jacobian that is restricted,

For H1 vector fields, want to ensure we have continuity at the nodes from either
side of the skeleton mesh that corresponds to the interface of charts
(i.e. where panel_ids differ on either side)

"""

module SkeletonTriangulationTests

using GridapGeosciences
using Gridap
using Gridap.Helpers
using Gridap.Geometry
import GridapGeosciences.Geometry: get_cell_ambient_maps

## Velocity field in tangent space of sphere
uX(x) = VectorValue(x[1]*x[3], x[2]*x[3], x[3]^2 - 1)

n_ref_lvls = 1
radius = 1.0
models = generate_refined_models(n_ref_lvls, CubedSphereMesh(radius), IntrinsicManifold())
atlas_model = models[end]
model = atlas_model.model # extract model to remove any bugs with adaptivity


Ω = Triangulation(model)
ambient_map_cf = AmbientMapCellField(Ω)
covariant_basis_cf = transpose∘∇(ambient_map_cf)
u_contra_cf = (pinvJ∘covariant_basis_cf)⋅(uX∘ambient_map_cf)

reffe_u  = ReferenceFE(lagrangian,VectorValue{2, Float64},2)
V = TestFESpace(Ω, reffe_u; conformity=:H1)
U = TrialFESpace(V)

uh = interpolate(u_contra_cf,U)

## Push uh to the ambient space
u_ambient = covariant_basis_cf⋅uh

## restrict to plus and minus side (in the ambient space)
u_tilde_plus = (u_ambient).plus
u_tilde_minus = (u_ambient).minus


###### get a mask that is the interface of charts
topo = get_grid_topology(model)
Dc = num_cell_dims(topo)
e2c = Gridap.Geometry.get_faces(topo,1,Dc)
panel_ids = get_cell_ambient_maps(model).ptrs

mask = zeros(num_facets(model))
for (i,edge) in enumerate(e2c)
  pid_1 = panel_ids[edge[1]]
  pid_2 = panel_ids[edge[2]]
  if pid_1 != pid_2
    mask[i] = 1
  end
end


##### Skeleton mesh using the mask
Λ = SkeletonTriangulation(model,Bool.(mask))
pts_plus = get_cell_points(Λ.plus)
pts_minus = get_cell_points(Λ.minus)
pts = get_cell_points(Λ)

### restrict to the plus and minus side in the chart, and push with a Jacobian that is restricted,
ambient_map_cf = AmbientMapCellField(Λ)
J_plus = transpose∘∇(ambient_map_cf.plus)
J_minus = transpose∘∇(ambient_map_cf.minus)
u_skel_plus = J_plus⋅(uh.plus)
u_skel_minus = J_minus⋅(uh.minus)

### Check that the restriction of the ambient solution to the plus side (u_tilde_plus)
### is equivanelt to J.plus ⋅ u.plus. And same for minus side
@check all(u_tilde_plus(pts_plus) .≈ u_skel_plus(pts_plus))
@check all(u_tilde_minus(pts_minus) .≈ u_skel_minus(pts_minus))

### Check the jump of the ambient solution and the skeleton solution is zero
@check all( u_tilde_plus(pts_plus) .≈ u_tilde_minus(pts_minus))
@check all( u_skel_plus(pts_plus) .≈ u_skel_minus(pts_minus))

### Check the continuity of the normal vector
n_tilde = pushforward_reference_normal(Λ)
n_tilde_plus = n_tilde.plus
n_tilde_minus = n_tilde.minus
@check all((n_tilde_plus)(pts) .≈ -(n_tilde_minus)(pts))


end  # module
