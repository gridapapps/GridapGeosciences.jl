
using GridapGeosciences
using Gridap
using Gridap.Helpers
 using Gridap.Geometry
import GridapGeosciences.Geometry: get_cell_ambient_maps

## The pressure field is zeromean
uX(x) = VectorValue(x[1]*x[3], x[2]*x[3], x[3]^2 - 1)

n_ref_lvls = 2
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

##### START SKELETON
Λ = SkeletonTriangulation(model)
pts_plus = get_cell_points(Λ.plus)
pts_minus = get_cell_points(Λ.minus)

ambient_map_cf = AmbientMapCellField(Λ)
J_plus = transpose∘∇(ambient_map_cf.plus)
J_minus = transpose∘∇(ambient_map_cf.minus)

## Compute the plus solution with the plus jacobian (same for minus sol)
u_skel_plus = J_plus⋅(uh.plus)
u_skel_minus = J_minus⋅(uh.minus)

### Check that the restriction of the ambient solution to the plus side (u_tilde_plus)
### is equivanelt to J.plus ⋅ u.plus
plus_out = (u_tilde_plus - u_skel_plus)(pts_plus)
sum(map(x->norm(x),plus_out))

### However, not the same for minus
minus_out = (u_tilde_minus - u_skel_minus)(pts_minus) ## some of these values are not zero

### Check the jump of the ambient solution is zero
_plus_out = u_tilde_plus(pts_plus) - u_tilde_minus(pts_minus)
sum(map(x->norm.(x),_plus_out))

### See that the jump of the skeleton solution is not zero
u_skel_plus(pts_plus) - u_skel_minus(pts_minus)
