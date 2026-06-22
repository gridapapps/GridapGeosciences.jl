"""
Test the pushforward of normal vectors, pullback of area form, and continuity
of metric on skeleton mesh
"""

module DistributedNormalTests

using Gridap
using GridapDistributed
using GridapGeosciences
using Test

################################################################################
#### Test unit normal vectors
################################################################################
myisless(b::Gridap.TensorValues.MultiValue,a::Number) = all(Gridap.TensorValues.isless.(b.data,a))

function test_debug_vector_equality(out,tol=1e-12)
  map(out) do o
    @test all( lazy_map(x-> all(myisless.(x,tol)), o))
  end
end

function test_debug_equality(out,tol=1e-12)
  map(out) do o
    @test all( lazy_map(x-> all(isless.(x,tol)), o))
  end
end

function main(distribute,nprocs)

  ranks = distribute(LinearIndices((nprocs,)))

  n_ref_lvls = 2
  radius = 1.0
  dmodels = get_distributed_intrinsic_cubed_sphere_refined_models(ranks,n_ref_lvls,radius)
  panel_model = dmodels[2]

  Ω_panel = Triangulation(panel_model)
  Λ = SkeletonTriangulation(with_ghost,panel_model)
  n_Λ = get_normal_vector(Λ)
  pts = get_cell_points(Λ)
  ##############################################################################
  ## Face normals: skeleton triangulation
  ##############################################################################

  # Method 1: Use gridap machinery
  n = pushforward_reference_normal(Λ)
  out = (n.plus+n.minus)(pts)
  test_debug_vector_equality(out)

  # Method 2: Santi's formula
  panel_ids = get_panel_ids(panel_model)
  n = pushforward_parametric_normal(Λ)
  out = (n.plus+n.minus)(pts)

  # test_debug_vector_equality(out) #### For some reason this is failing, I am unsure why
  #### Test the equality of n.plus and n.minus using local evaluation
  map(local_views(Λ),local_views(n.plus),local_views(n.minus)) do strian,cfplus,cfminus
    strian_pts = get_cell_points(strian)
    o = cfplus(strian_pts) + cfminus(strian_pts)
    @test all( lazy_map(x-> all(myisless.(x,1e-12)), o))
  end


  ##############################################################################
  ## DG tests
  ### check sqrt(g) is continuous across skeleton
  ### check |Jg^-1 n| - pullback of area form
  ##############################################################################
  meas_cf = MeasureCellField(Λ)
  out = (meas_cf.plus-meas_cf.minus)(pts)
  test_debug_equality(out)

  area_form_cf = pullback_area_form(Λ)
  out = (area_form_cf.plus-area_form_cf.minus)(pts)
  test_debug_equality(out)

  ##############################################################################
  ## Advection tests
  ### check abs(v⋅n.plus) = abs(v⋅n.minus)
  ##############################################################################
  vecX(XYZ) = VectorValue(-XYZ[2],XYZ[3],0.0)
  vX = tangent_vec(vecX)

  V = TestFESpace(panel_model, ReferenceFE(raviart_thomas,Float64,1); conformity=:HDiv)
  U = TrialFESpace(V)

  ambient_map_cf = AmbientMapCellField(Ω_panel)
  covariant_basis_cf = transpose∘∇(ambient_map_cf)
  _vel = (pinvJ∘covariant_basis_cf)⋅(vX∘ambient_map_cf)
  vel = interpolate(_vel,U)

  diff_cf = (abs((vel⋅ n_Λ).minus) .- abs((vel⋅ n_Λ).plus))(pts)
  test_debug_equality(diff_cf)
end

end # module
