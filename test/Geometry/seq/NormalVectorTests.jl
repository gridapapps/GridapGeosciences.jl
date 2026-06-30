"""
In this module, we test the pushforward of normal vectors from the parametric
to ambient space, for boundary and skeleton triangulations
"""

module NormalTests

using Gridap
using GridapGeosciences
using Test
using Gridap.Geometry
using Gridap.Helpers


################################################################################
#### Test unit normal vectors
################################################################################
function normal_vector_from_basis(J)
    a1 = VectorValue(J[1],J[2],J[3])
    a2 = VectorValue(J[4],J[5],J[6])
    n = cross(a1,a2)
    _n = n*(1/sqrt(det(J'⋅J)) )
    @check Gridap.TensorValues.meas(_n) ≈ 1.0
    _n
end

function test_normal_unit_vector(atlas_model,return_vtk=false)
  lvl = nref(nc(atlas_model))
  println("nref = $lvl")

  Ω_atlas = Triangulation(atlas_model)
  dΩ = Measure(Ω_atlas,6)

  ambient_map_cf = AmbientMapCellField(Ω_atlas)

  norm_vec_cf = sphere_surface_normal_vec ∘ ambient_map_cf
  norm_vec_from_basis_cf = normal_vector_from_basis∘transpose∘∇(ambient_map_cf)
  meas_cf = MeasureCellField(Ω_atlas)

  e  = norm_vec_cf-norm_vec_from_basis_cf
  e_l2 =  sqrt(sum(∫( (e⋅e)*meas_cf )dΩ))

  @test e_l2 < 1e-12

  if return_vtk
    lvl = nref(nc(atlas_model))
    panel_cfs = [ norm_vec_cf,norm_vec_from_basis_cf]
    labels = ["normal", "n"]
    cellfields = map((x,y) -> x=>y, labels,panel_cfs)
    writevtk_with_cell_geomap(AmbientMapCellField(Ω_atlas),Ω_atlas,dir*"/ambient_model_nref$(lvl)",cellfields=cellfields,append=false)
  end
end


n_ref_lvls = 4
radius = 1.0
models  = generate_refined_models(n_ref_lvls, CubedSphereMesh(radius), IntrinsicManifold())
return_vtk = false
dir = @__DIR__

################################################################################
## Unit normal to surface: k = a₁ × a₂
################################################################################
for atlas_model in models
  test_normal_unit_vector(atlas_model)
end

################################################################################
## Face normals -- for 1 model
################################################################################
atlas_model = models[4]

topo = get_grid_topology(atlas_model)
Dc = num_cell_dims(topo)
face_to_mask = get_isboundary_face(topo,Dc-1)
bgface_to_mask = collect(Bool, .!get_isboundary_face(topo,Dc-1))

################################################################################
## Face normals: boundary trian
################################################################################

trian = BoundaryTriangulation(atlas_model,bgface_to_mask,1)
pts = get_cell_points(trian)

# get face normals in 3D
n_3D = pushforward_reference_normal(trian)

# push forward 2D chart normals to ambient space
n_mapped = pushforward_parametric_normal(trian)

# test equality
@test all(n_mapped(pts) .≈ n_3D(pts))

################################################################################
## Face normals: skeleton trian
################################################################################
trian = SkeletonTriangulation(atlas_model)
pts = get_cell_points(trian)

# regular 2D normal in chart
n_2D = get_normal_vector(trian)

# # get face normals in 3D
n_3D = pushforward_reference_normal(trian)

# push forward 2D chart normals to ambient space
n_mapped = pushforward_parametric_normal(trian)

# test equality of plus and minus side
@test all(n_mapped.plus(pts) .≈ n_3D.plus(pts))
@test all(n_mapped.minus(pts) .≈ n_3D.minus(pts))

## plot normals on skeleton
if return_vtk
  panel_cfs = [n_3D.plus, n_3D.minus, n_3D.minus+n_3D.plus,
              n_2D.plus, n_2D.minus, n_2D.minus+n_2D.plus]
  labels = ["amb_n_plus", "amb_n_minus", "amb_n_total",
            "chart_n_plus", "chart_n_minus", "chart_n_total"]
  cellfields = map((x,y) -> x=>y, labels,panel_cfs)

  writevtk_with_cell_geomap(AmbientMapCellField(trian),trian,dir*"/ambient_model_skeleton",cellfields=cellfields,append=false)
end
################################################################################
## DG tests
### check sqrt(g) is continuous across skeleton
### show g, g^-1, J not continuous
### check |Jg^-1 n| - pullback of area form
################################################################################
Λ = SkeletonTriangulation(atlas_model)
pts = get_cell_points(Λ)
meas_cf = MeasureCellField(Λ)
inv_metric_cf = InvMetricCellField(Λ)
jac_cf_plus = ∇(AmbientMapCellField(Λ).plus)
jac_cf_minus = ∇(AmbientMapCellField(Λ).minus)
area_form_cf = pullback_area_form(Λ)

# test equality of plus and minus side of sqrt(g)
@test all(meas_cf.minus(pts) .≈ meas_cf.plus(pts))

# test equality of plus and minus side of |Jg^-1 n|
@test all(area_form_cf.plus(pts) .≈ area_form_cf.minus(pts))

# test equality of plus and minus side for g^-1
@test all(inv_metric_cf.plus(pts) .≈ inv_metric_cf.minus(pts))

# test inequality of plus and minus side for J
@test !all(jac_cf_minus(pts) .≈ jac_cf_plus(pts))

if return_vtk
  panel_cfs = [meas_cf.plus, meas_cf.minus, meas_cf.minus-meas_cf.plus,
              jac_cf.plus, jac_cf.minus, jac_cf.minus-jac_cf.plus,
              area_form_cf.plus, area_form_cf.minus, area_form_cf.plus-area_form_cf.minus]
  labels = ["g_plus", "g_minus", "g_diff", "jac_plus", "jac_minus", "jac_diff",
            "a_plus", "a_minus", "a_diff"]
  cellfields = map((x,y) -> x=>y, labels,panel_cfs)

  writevtk_with_cell_geomap(AmbientMapCellField(Λ),Λ,dir*"/ambient_model_skeleton",cellfields=cellfields,append=false)
end

################################################################################
## Advection tests
### check abs(v⋅n.plus) = abs(v⋅n.minus)
################################################################################
vecX(XYZ) = VectorValue(-XYZ[2],XYZ[3],0.0)
vX = sphere_tangent_vec_component(vecX)

V = TestFESpace(atlas_model, ReferenceFE(raviart_thomas,Float64,1); conformity=:HDiv)
U = TrialFESpace(V)

Ω_atlas = Triangulation(atlas_model)
Λ = SkeletonTriangulation(atlas_model)
n_Λ = get_normal_vector(Λ)
pts = get_cell_points(Λ)

ambient_map_cf = AmbientMapCellField(Ω_atlas)
covariant_basis_cf = transpose∘∇(ambient_map_cf)
_vel = (pinvJ∘covariant_basis_cf)⋅(vX∘ambient_map_cf)
vel = interpolate(_vel,U)

diff_cf = (abs((vel⋅ n_Λ).minus) .- abs((vel⋅ n_Λ).plus))
@test all(all.(lazy_map(x->isless.(x, 1e-14), diff_cf(pts))))

if return_vtk
  labels = ["upwind_plus","upwind_minus","upwind_diff"]
  panel_cfs = [abs((vel⋅ n_Λ).plus),abs((vel⋅ n_Λ).minus),abs((vel⋅ n_Λ).minus)-abs((vel⋅ n_Λ).plus)]
  cellfields = map((x,y) -> x=>y, labels,panel_cfs)
  writevtk_with_cell_geomap(AmbientMapCellField(Λ),Λ,dir*"/ambient_model_skeleton", cellfields=cellfields,append=false)
end

@test true

end #module
