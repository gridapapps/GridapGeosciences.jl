"""
solve vector laplacian in mixed form
σ + ∇ᵧ⋅ϕ  = 0
∇ᵧ × (∇ᵧ × ϕ) + ∇ᵧσ = f
where f = -Δϕ
"""

module HodgeLaplacianVectorTests

using Gridap
using Gridap.Helpers
using Gridap.Algebra
using GridapDistributed
using GridapGeosciences
using GridapP4est
using Test


inv_jacobian(p) = x -> inv(forward_jacobian(p)(x))
contra_v_3D(vecX::Function,p) = x -> inv_jacobian(p)(x) ⋅ vecX(p)(x)
contra_v_3D(vecX::Function) = p -> contra_v_3D(vecX,p)

transpose_jacobian(p) = x -> transpose(forward_jacobian(p)(x))
inv_tranpose_jacobian(p) = x -> inv(transpose_jacobian(p)(x))

function uX(forward_map)
  function _u(γαβ)
    xyz = forward_map(γαβ)
    # VectorValue(-xyz[2],xyz[1],0.0)

    r = sqrt(xyz[1]^2 + xyz[2]^2 + xyz[3]^2)
    f = 2.0*xyz[3]/r
    n = normal_vec(xyz)
    f*n
  end
end



function hodge_laplacian_vector(
  panel_model::GridapDistributed.GenericDistributedDiscreteModel{3,3},
  p_fe::Int,dir::String,uX::Function,ls=LUSolver(),return_vtk=false;
  _i_am_main=true)

  Dc = num_cell_dims(panel_model)
  lvl = nref(panel_model)
 _i_am_main && println("p_fe = $(p_fe); nref = $lvl; Dc = $Dc")

  # degree = 30
  degree = 5*(p_fe + 1)
  if p_fe == 0
    degree = 10
  end
  @check degree > 0 "Zero quad!!"

  ## finite element solver
  Ω_panel = Triangulation(panel_model)
  dΩ = Measure(Ω_panel,degree)
  Ω_error = Triangulation(panel_model)
  dΩ_error = Measure(Ω_error,2*degree)

  tags = ["top_boundary", "bottom_boundary"]
  Γ = BoundaryTriangulation(panel_model,tags=tags)
  dΓ = Measure(Γ,degree)
  nΓ = get_normal_vector(Γ)

  ## metric information
  inv_metric_cf = ParametricCellField(inv_metric,Ω_panel)
  metric_cf = ParametricCellField(metric,Ω_panel)
  meas_cf = ParametricCellField(sqrtg,Ω_panel)
  covariant_basis_cf = ParametricCellField(covariant_basis,Ω_panel)



  # covarient components of u
  function ucov(forward_map)
    function _u(γαβ)
      u = uX(forward_map)(γαβ)
      J = transpose_jacobian(forward_map)(γαβ)
      J⋅u
    end
  end

  # u ⋅ n on the surface
  function unX(forward_map)
    function _u(γαβ)
      xyz = forward_map(γαβ)
      u = uX(forward_map)(γαβ)
      n = normal_vec(xyz)
      u⋅n
    end
  end

  ### Curl of covariant components of u
  curlu(p,x) = curl(ucov(p))(x)
  curlu(p) = x -> curlu(p,x)
  _curlu(p) = curl(ucov(p))

  ### Covariant components of surfcurl u
  wcov(p,x) = 1/sqrtg(p,x)*metric(p,x)⋅curlu(p,x)
  wcov(p) = x -> wcov(p,x)
  curlw(p,x) = curl(wcov(p))(x)
  curlw(p) = x -> curlw(p,x)

  #### Covariant component of surfcurl surfcurl u
  curlw_cov(p) = x -> 1/sqrtg(p,x)*metric(p,x)⋅curlw(p,x)

  # area measure
  _area_meas(p) = x->  forward_jacobian(p,x) ⋅ (inv_metric(p,x) ⋅ VectorValue(1,0,0))
  area_meas(p) = x-> norm(_area_meas(p)(x))

  #### Covariant componetsn of (surfcurl u)× surfnormal
  wcrossk_cov(p) = x -> 1/sqrtg(p,x) * metric(p,x)⋅(wcov(p,x) × (VectorValue(1,0,0)/area_meas(p)(x)) )


  # _t(p) = x -> sqrtg(p)(x)* contra_v_3D(uX,p)(x)
  # _k(p) = x -> 1/sqrtg(p)(x) * ( divergence(_t(p) )(x) )
  # _l(p) = gradient(_k(p))
  # rhs(p) = x-> curlw_cov(p)(x) - _l(p)(x)

  ## surface divergence
  _sdiv_u(p) = x -> sqrtg(p)(x)* contra_v_3D(uX,p)(x)
  sdiv_u(p) = x -> 1/sqrtg(p)(x) * ( divergence(_sdiv_u(p) )(x) )

  ### covariant components of surfgrad(surfdiv u)
  graddiv_cov(p) = gradient(sdiv_u(p))

  ### Rhs function
  rhs(p) = x-> curlw_cov(p)(x) - graddiv_cov(p)(x)
  rhs_cov_cf = ParametricCellField(rhs,Ω_panel)

  u_cov_cf = ParametricCellField(ucov,Ω_panel)
  ccurlu_cov_cf = ParametricCellField(curlw_cov,Ω_panel)
  un_cf = ParametricCellField(unX,Ω_panel)
  curlu_cross = ParametricCellField(wcrossk_cov,Ω_panel)
  curlu_cf = ParametricCellField(curlu,Ω_panel)

  sdiv_cf =  ParametricCellField(surfdiv(contra_v_3D(uX)),Ω_panel)
  sigma_cf = -sdiv_cf


  # cellfields = ["curlu"=>ccurlu_cov_cf,
  #               "u"=>covariant_basis_cf ⋅ (inv_metric_cf⋅u_cov_cf),
  #               "un"=>un_cf,
  #               "curlu_cross"=>covariant_basis_cf ⋅ (inv_metric_cf⋅curlu_cross),
  #               "sigma"=>-sdiv_cf,
  #               "rhs"=>rhs_cov_cf
  #               ]
  # writevtk_with_cell_geomap(geo_map_func(Ω_panel),Ω_panel,dir*"/sol",
  #         cellfields=cellfields,
  #         append=false)


  ## FE spaces
  T = TestFESpace(Ω_panel, ReferenceFE(lagrangian,Float64,p_fe+1); conformity=:H1)
  S = TrialFESpace(T)

  R = TestFESpace(Ω_panel, ReferenceFE(nedelec,Float64,p_fe);conformity=:Hcurl)
  H = TrialFESpace(R)

  sigma_int = interpolate(sigma_cf,S)
  u_int = interpolate(u_cov_cf,H)

  ### Multifield
  X = MultiFieldFESpace([S,H])
  Y = MultiFieldFESpace([T,R])

  biform_x((s,u),(t,v)) = (
                  ∫( (s*t)*meas_cf  )dΩ
                - ∫( ∇(t)⋅(inv_metric_cf⋅u)*meas_cf  )dΩ
                + ∫( curl(u)⋅(metric_cf⋅curl(v))*(1/meas_cf) )dΩ
                + ∫( gradient(s)⋅(inv_metric_cf⋅v)*meas_cf )dΩ
                  )
  liform_x((t,v)) = (
                ∫( rhs_cov_cf⋅(inv_metric_cf⋅v)*meas_cf  )dΩ
                + ∫( v⋅( ( metric_cf⋅curlu_cf )×nΓ    )*(1/meas_cf)     )dΓ
                - ∫(( t*(u_cov_cf⋅(inv_metric_cf⋅nΓ)) )*(meas_cf)  )dΓ
                  )


  op = AffineFEOperator(biform_x,liform_x,X,Y)
  xh = solve(ls,op)
  sh,uh = xh

  # final_dir = dir*"/final_solution"
  # # ensure no MPI task tries to generate the file before the main MPI task has
  # # created the folder
  # PartitionedArrays.barrier(ranks)
  # psave(final_dir*"/sol",xh)

  # _e = sigma_cf - sh
  _e = sigma_int - sh
  el2_s = sqrt(sum(∫( (_e*_e)*meas_cf  )dΩ_error))

  # _e = (inv_metric_cf⋅uh) - (inv_metric_cf⋅u_cov_cf)
  _e = (inv_metric_cf⋅uh) - (inv_metric_cf⋅u_int)
  el2_u = sqrt(sum(∫( (_e⋅(metric_cf ⋅_e))*meas_cf  )dΩ_error))

 _i_am_main && println("eu = $(el2_u), es = $(el2_s)")

  if return_vtk
    cellfields =  ["u"=>covariant_basis_cf ⋅ (inv_metric_cf⋅u_cov_cf),
    "uh"=>covariant_basis_cf ⋅ (inv_metric_cf⋅uh),
    "eu"=>covariant_basis_cf ⋅ (inv_metric_cf⋅uh)-covariant_basis_cf ⋅ (inv_metric_cf⋅u_cov_cf),
    "sh"=>sh, "s"=>sigma_cf, "e"=>sh-sigma_cf
                  ]
    writevtk_with_cell_geomap(geo_map_func(Ω_panel),Ω_panel,dir*"/ambient_model_nref$(lvl)_p$p_fe",
            cellfields=cellfields,append=false)
  end


  return el2_u, el2_s, false

end


################################################################################
#### Auto convergence test -- only 3D models for p=[0,1]
################################################################################
function main(models::AbstractArray;ps = [1],_i_am_main=true)

  ls = LUSolver()
  dir = @__DIR__
  p_convergence_auto_test(ps,models,hodge_laplacian_vector,dir,uX,ls;_i_am_main=_i_am_main)
end



end # module
