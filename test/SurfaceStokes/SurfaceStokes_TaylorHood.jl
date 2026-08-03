""" Solve the vector surface Stokes + Laplacian in 2D
  -ν Δᵧ(̃u) + α ̃u + ∇ᵧ ̃p = ̃f
                  ∇ᵧ⋅̃u  = 0
where Δᵧ(̃u) = ∇ᵧ(∇ᵧ⋅̃u) - ∇ᵧ^⟂(∇ᵧ^⟂ ⋅ ̃u)
The velocity is in H1 vector-valued lagrangian, order_u ≥ 2
The pressure is in H1 continuous lagrangian, order_p = order_u - 1
Using the Taylor--Hood pair
"""

using GridapGeosciences
using Gridap
using Gridap.Helpers
using DrWatson

import GridapGeosciences.CellData: deriv_det, deriv_sqrt, cpAB

include("operator.jl")

## Need to use velocity field that is tangent, but not divergence free. Thus, the
## incompressibility condition div u = 0 does not hold, and the Stokes system
## must be solved.
## The pressure field is zeromean
uX(x) = VectorValue(x[1]*x[3], x[2]*x[3], x[3]^2 - 1)
pX(x) = x[3]

n_ref_lvls = 2
radius = 1.0
models = generate_refined_models(n_ref_lvls, CubedSphereMesh(radius), IntrinsicManifold())
atlas_model = models[end]
p_fe = 2
ls = LUSolver()
_i_am_main = true

# function surface_stokes(atlas_model,
#   p_fe::Int,dir::String,uX::Function,pX::Function,ls=LUSolver(),return_vtk=false;
#   _i_am_main=true)

  α = 1.0
  ν = 1.0

  Dc = num_cell_dims(atlas_model)
  lvl = nref(atlas_model)
 _i_am_main && println("p_fe = $(p_fe); nref = $lvl; Dc = $Dc")

  @check p_fe >= 2 "/n # Taylor hood pair requires p≥2 "

  degree = 6*(p_fe+1)
  println("degree = ", degree)
  Ω_atlas = Triangulation(atlas_model)
  dΩ = Measure(Ω_atlas,degree)
  dΩ_error = Measure(Ω_atlas,3*degree)

  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  metric_cf = MetricCellField(Ω_atlas)
  inv_metric_cf = InvMetricCellField(Ω_atlas)
  meas_cf = MeasureCellField(Ω_atlas)
  covariant_basis_cf = transpose∘∇(ambient_map_cf)

  grad_meas_cf = (deriv_sqrt∘det∘metric_cf)*Operation(cpAB)(deriv_det∘metric_cf,gradient(metric_cf))

  perp_metric_cf = PerpMetric(Ω_atlas)

  ## Expanded product rule to help with weak form
  # div ( √g u) = div(u)*√g + u⋅gradient(√g)
  divg_product_rule(u) = divergence(u)*meas_cf + u⋅grad_meas_cf

  ## Manufactured solution
  p_cf = pX∘ambient_map_cf
  u_contra_cf = (pinvJ∘covariant_basis_cf)⋅(uX∘ambient_map_cf)

  sum(∫( p_cf  )dΩ) # check zeromean

  sigma_cf = divs(uX,Ω_atlas) # div u to add to pressure equation
  rhs_curl_cf = vecΔs_2D(uX, Ω_atlas)
  rhs = -1.0*ν*rhs_curl_cf + α*u_contra_cf + ∇s_contra(pX,Ω_atlas)
  # rhs = -1.0*ν*rhs_curl_cf + α*u_int + inv_metric_cf⋅(gradient(p_int))

  ## FE spaces: Taylor hood pair --> Q2/Q1 continuous
  reffe_u  = ReferenceFE(lagrangian,VectorValue{2, Float64},p_fe)
  reffe_p = ReferenceFE(lagrangian,Float64,p_fe-1)

  V = TestFESpace(Ω_atlas, reffe_u; conformity=:H1)
  U = TrialFESpace(V)

  Q = TestFESpace(Ω_atlas, reffe_p; conformity=:H1,constraint=:zeromean)
  P = TrialFESpace(Q)

  Y = MultiFieldFESpace([V, Q])
  X = MultiFieldFESpace([U, P])

  # Interpolate the exact solution into the FE space
  p_int = interpolate(p_cf,P)
  u_int = interpolate(u_contra_cf,U)

  # ## Weak form
  biform_curl((u,p),(v,q)) = ( ∫( ν*(divg_product_rule(u)*divg_product_rule(v))*(1/meas_cf) )dΩ
                            +  ∫( -1.0*ν*(divergence(perp_metric_cf⋅u)*divergence(perp_metric_cf⋅v))*(1/meas_cf)  )dΩ
                            )

  biform_u((u,p),(v,q)) = ∫( α*((u⋅(metric_cf⋅v))*meas_cf)  )dΩ - ∫( p*divg_product_rule(v)  )dΩ

  biform_p((u,p),(v,q)) = ∫( q*divg_product_rule(u)  )dΩ

  biform((u,p),(v,q)) = biform_curl((u,p),(v,q)) + biform_u((u,p),(v,q)) + biform_p((u,p),(v,q))


  # the manufactured solution is exactly the LHS operator
  #  liform((v,q)) = (
  #    ∫( (-1.0*ν*rhs_curl_cf⋅(metric_cf⋅v))*meas_cf  )dΩ
  #   +  ∫( (u_int⋅ (metric_cf⋅v))*(meas_cf) )dΩ
  #   + ∫( (gradient(p_int)⋅v)*meas_cf )dΩ # assume regularity to IBP
  #   + ∫( q*divg_product_rule(u_int) )dΩ
  #   )

  liform((v,q)) = ∫( (rhs⋅(metric_cf⋅v))*meas_cf  )dΩ + ∫( (sigma_cf*q)*meas_cf )dΩ

  # ## FE problem
  op = AffineFEOperator(biform,liform,X,Y)
  A = get_matrix(op)
  b = get_vector(op)
  ns = numerical_setup(symbolic_setup(ls,A),A)
  x = Gridap.Algebra.allocate_in_domain(A); fill!(x,0.0)
  solve!(x,ns,b)
  xh = FEFunction(X,x)
  uh,ph = xh
  # uh = FEFunction(U,rand(num_free_dofs(U)))


  # For the depth, the $L^2$ norm of the error between the exact and numerical solutions is computed as
  ep = p_int - ph
  ep_l2 = sqrt(sum(∫((ep⋅ep)*meas_cf)dΩ_error))

  # For the velocity, the $L^2$ norm of the error between
  # the exact and numerical solutions is computed as
  eu = u_int - uh
  eu_l2 = sqrt(sum(∫( eu⋅(metric_cf⋅eu)*(meas_cf) )dΩ_error))
  eu_h1 = sqrt(sum(∫( eu⋅(metric_cf⋅eu)*(meas_cf) + (gradient(eu)⊙(inv_metric_cf⋅gradient(eu)))*meas_cf  )dΩ_error))

  ######### START JUMP CALCS
  ### The jump of u on the surface is:
  ## [̃u] = ( J u ⋅ J g^{-1} n/|| J g^{-1} n ||  ).plus  + ( J u ⋅ J g^{-1} n/|| J g^{-1} n ||  ).minus
  ##     =  (u⋅n/|| J g^{-1} n || ).plus  + (u⋅n/|| J g^{-1} n || ).mins

  using Gridap.Geometry
  import GridapGeosciences.Geometry: get_cell_ambient_maps

  # get a mask that is the interface of charts
  topo = get_grid_topology(atlas_model)
  Dc = num_cell_dims(topo)
  c2e = Gridap.Geometry.get_faces(topo,Dc,1)
  e2c = Gridap.Geometry.get_faces(topo,1,Dc)
  panel_ids = get_cell_ambient_maps(atlas_model.model).ptrs

  mask = zeros(num_facets(atlas_model))
  for (i,edge) in enumerate(e2c)
    pid_1 = panel_ids[edge[1]]
    pid_2 = panel_ids[edge[2]]
    if pid_1 != pid_2
      mask[i] = 1
    end
  end

  # Skeleton triangulation
  skel = SkeletonTriangulation(atlas_model,Bool.(mask))
  num_cells(skel)


  # For simplicity, extract the trian out of the adapted trian
  Λ = skel.trian
  dΛ = Measure(Λ,6)#3*degree)
  nΛ = get_normal_vector(Λ)
  pts_skel = get_cell_points(dΛ)
  area = pullback_area_form(Λ)

  # Compute the jump on quad points, it is zero
  (  (uh.plus ⋅ nΛ.plus)/area.plus + (uh.minus ⋅ nΛ.minus)/area.minus )(pts_skel)


  ## Compute the full jump with all the terms, see it is also zero
  ambient_map_cf = AmbientMapCellField(Λ)
  J_plus = transpose∘∇(ambient_map_cf.plus)
  J_minus = transpose∘∇(ambient_map_cf.minus)
  ginv = InvMetricCellField(Λ)

  ( ( (J_plus⋅ uh.plus) ⋅ (J_plus ⋅(ginv.plus⋅ nΛ.plus)) )/area.plus
      + ( (J_minus⋅ uh.minus) ⋅ (J_minus ⋅(ginv.minus⋅ nΛ.minus)) )/area.minus)(pts_skel)


  ### Another approach: move the FE function to the plus side of the trian, and minus
  ### sides using the f2c reference map. Then evaluate at cell_points
  trian = Λ.plus
  pts_plus = get_cell_points(trian)
  glue = trian.glue
  bgmodel = get_background_model(trian)
  cell_grid = get_grid(bgmodel)
  face_grid = get_grid(trian)
  f2c_ref_map = Gridap.Geometry.compute_face_to_cell_reference_map(cell_grid,face_grid,glue)
  plus_cf = lazy_map(∘,get_data(uh),f2c_ref_map)
  plus = Gridap.CellData.GenericCellField(plus_cf, trian, ReferenceDomain())

  trian = Λ.minus
  pts_minus = get_cell_points(trian)
  glue = trian.glue
  bgmodel = get_background_model(trian)
  cell_grid = get_grid(bgmodel)
  face_grid = get_grid(trian)
  f2c_ref_map = Gridap.Geometry.compute_face_to_cell_reference_map(cell_grid,face_grid,glue)
  minus_cf = lazy_map(∘,get_data(uh),f2c_ref_map)
  minus = Gridap.CellData.GenericCellField(minus_cf, trian, ReferenceDomain())

  # compute the jump, this one is not zero!
  (minus⋅nΛ.minus/area.minus)(pts_minus) + (plus⋅nΛ.plus/area.plus)(pts_plus)







  #### OLD!!


  function change_fe_func_to_skel(fe_func, Λ)
    cdata_plus = change_domain(fe_func, Λ.trian.plus,DomainStyle(fe_func))
    plus =  Gridap.CellData.GenericCellField(get_data(cdata_plus), Λ, Gridap.CellData.DomainStyle(fe_func))

    cdata_minus = change_domain(fe_func, Λ.trian.minus,DomainStyle(fe_func))
    minus =  Gridap.CellData.GenericCellField(get_data(cdata_minus), Λ, Gridap.CellData.DomainStyle(fe_func))

    fe_func_skel = Gridap.CellData.SkeletonCellFieldPair(plus,minus)
    fe_func_skel
  end

  eu_skel = change_fe_func_to_skel(eu,Λ)
  uh_skel = change_fe_func_to_skel(uh,Λ)



  inv_metric_cf_skel = InvMetricCellField(Λ).plus
  metric_cf_skel = MetricCellField(Λ).plus
  meas_cf_skel = MeasureCellField(Λ).plus
  area_form_cf_skel = pullback_area_form(Λ).plus
  ambient_map_cf = AmbientMapCellField(Λ).plus
  covariant_basis_cf_skel = transpose∘∇(ambient_map_cf)

  # dir = @__DIR__
  #  writevtk_with_cell_geomap(AmbientMapCellField(Λ),Λ,dir*"/jumps",
  #         cellfields=["jump"=>jump( covariant_basis_cf_skel ⋅ uh_skel)],append=false)

  γ = 1.0
  dxx = dx(atlas_model)

  bulk_term = sum(∫( (gradient(eu_skel)⊙(inv_metric_cf_skel⋅gradient(eu_skel)))*( area_form_cf_skel*meas_cf_skel  )  )dΛ)
  # jump_term =  (sum(∫( (jump(uh_skel)⋅(metric_cf_skel ⋅ jump(uh_skel)))*( area_form_cf_skel*meas_cf_skel ) )dΛ))


  function jump_u(uh,nΛ,area)
    (uh.plus⋅nΛ.plus)/area.plus + (uh.minus⋅nΛ.minus)/area.minus
  end

  jump_term = (sum(∫( ( jump_u(uh_skel,nΛ,area_skel)*jump_u(uh_skel,nΛ,area_skel)  )*(  meas_cf_skel*area_form_cf_skel ) )dΛ))
dir = @__DIR__
   writevtk_with_cell_geomap(AmbientMapCellField(Λ),Λ,dir*"/jumps",
          cellfields=["plus"=>plus, "minus"=>minus,
          "dif"=> plus - minus, "jump"=>plus+minus],append=false)





  sum( ∫( ( (plus+minus)*(plus+minus)   ) )dΛ )




  # jump_term =  (sum(∫( (jump(covariant_basis_cf_skel⋅uh_skel)⋅jump(covariant_basis_cf_skel⋅uh_skel))*( area_form_cf_skel*meas_cf_skel ) )dΛ))
#

 _i_am_main && println("eu = $(eu_l2), ep = $(ep_l2), euh1 = $eu_h1")
  _i_am_main && println("bulk= $(bulk_term), jump = $(jump_term)")


  if return_vtk
      panel_cfs = [p_int,ph,ep, covariant_basis_cf⋅ u_int, covariant_basis_cf⋅ uh,eu]
      labels = ["p","ph", "ep", "u","uh","eu"]
      cellfields = map((x,y) -> x=>y, labels,panel_cfs)
      writevtk_with_cell_geomap(AmbientMapCellField(Ω_atlas),Ω_atlas,dir*"/stokes_nref$(lvl)_p$p_fe",
          cellfields=cellfields,append=false)
  end


  n = num_cells(atlas_model)/6
  n_ref = lvl
  output = @strdict eu_l2 eu_h1 ep_l2 n p_fe n_ref Dc bulk_term jump_term
  safesave(datadir(dir*"/convergence", ("stokes_nref$(n_ref)_p$(p_fe)_D$Dc.jld2")), output)



  return eu_l2, ep_l2, false

end


function main(models::AbstractArray;ps=[2,3,4],_i_am_main=true)
  # Taylor hood pair requires p≥2
  ls = LUSolver()
  dir = @__DIR__
  p_convergence_auto_test(ps,models,surface_stokes,dir,uX,pX,ls;_i_am_main=_i_am_main)
end

n_ref_lvls = 4
radius = 1.0
models = generate_refined_models(n_ref_lvls, CubedSphereMesh(radius), IntrinsicManifold())


using GridapPETSc
options = """
          -ksp_type preonly -ksp_error_if_not_converged true
          -pc_type lu -pc_factor_mat_solver_type mumps
          -mat_mumps_icntl_1 4
          -mat_mumps_icntl_4 0
          -mat_mumps_icntl_7 0
          -mat_mumps_icntl_14 100
          -mat_mumps_icntl_28 1
          -mat_mumps_icntl_29 2
          -mat_mumps_cntl_3 1.0e-6
          """
GridapPETSc.Init(args=split(options))
ls = GridapPETSc.PETScLinearSolver()

# p_fe = 2
# ls = LUSolver()
# ls = BackslashSolver()
for p_fe in [2]
  for model in models
    surface_stokes(model,
      p_fe,@__DIR__,uX,pX,ls,true;_i_am_main=true)
  end
end
