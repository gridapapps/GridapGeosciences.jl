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
  ## Face normals: skeleton trian
  ##############################################################################

  # Method 1: Use gridap machinary
  n = pushforward_normal(Λ)
  out = (n.plus+n.minus)(pts)
  test_debug_vector_equality(out)

  # Method 2: Santi's formula
  panel_ids = get_panel_ids(panel_model)
  forward_map_generator = get_forward_map_generator(panel_model)
  cell_geo_map = geo_map_func(forward_map_generator,panel_ids)
  n = pushforward_normal(Λ,cell_geo_map)
  out = (n.plus+n.minus)(pts)

  # test_debug_vector_equality(out) #### For some reason this is failing, I am unsure why
  #### Test the equality of n.plus and n.minus using local evaluation
  map(local_views(Λ),local_views(n.plus),local_views(n.minus)) do strian,cfplus,cfminus
    plus = strian.plus
    pts_plus = get_cell_points(plus)

    minus = strian.minus
    pts_minus = get_cell_points(minus)

    o = cfplus(pts_plus) + cfminus(pts_minus)
    @test all( lazy_map(x-> all(myisless.(x,1e-12)), o))
  end


  ##############################################################################
  ## DG tests
  ### check sqrt(g) is continuous across skeleton
  ### check |Jg^-1 n| - pullback of area form
  ##############################################################################
  meas_cf = ParametricCellField(sqrtg,Λ)
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
  vX = panel_to_cartesian(tangent_vec(vecX))

  V = TestFESpace(panel_model, ReferenceFE(raviart_thomas,Float64,1); conformity=:HDiv)
  U = TrialFESpace(V)

  _vel = ParametricCellField(contra_v(vX),Ω_panel)
  vel = interpolate(_vel,U)

  diff_cf = (abs((vel⋅ n_Λ).minus) .- abs((vel⋅ n_Λ).plus))(pts)
  test_debug_equality(diff_cf)

end


end # module
