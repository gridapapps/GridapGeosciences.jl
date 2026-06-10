transpose_jacobian(p) = x -> transpose(forward_jacobian(p)(x))
covar_v_3D(vecX::Function,m) = αβ -> transpose_jacobian(m)(αβ) ⋅ vecX(m)(αβ)
covar_v_3D(vecX::Function) = m -> covar_v_3D(vecX,m)

function L2_projection_Hcurl(panel_model,
                             p_fe::Int,
                             dir::String,
                             vecX::Function,
                             ls=LUSolver(),
                             return_vtk=true;
                             _i_am_main=true)

  Dc = num_cell_dims(panel_model)
  lvl = nref(panel_model)

  @check Dc == 3

  degree = 4*(p_fe+1)
  if p_fe == 0
    degree = 8
  end

  _i_am_main && println("p_fe = $(p_fe); nref = $lvl; Dc = $Dc; degree = $(degree)")

  Ω_atlas = Triangulation(panel_model)
  dΩ = Measure(Ω_atlas,degree)
  dΩ_error = Measure(Ω_atlas,2*degree)

  inv_metric_cf = ParametricCellField(inv_metric,Ω_atlas)
  metric_cf = ParametricCellField(metric,Ω_atlas)
  meas_cf = ParametricCellField(sqrtg,Ω_atlas)
  covariant_basis_cf = ParametricCellField(covariant_basis,Ω_atlas)

  ## covariant components
  vec_cov_cf = ParametricCellField(covar_v_3D(vecX),Ω_atlas)
  vec_proj_cf = covariant_basis_cf⋅ ( inv_metric_cf ⋅ vec_cov_cf)


  reffe = ReferenceFE(nedelec,Float64,p_fe)
  V = TestFESpace(panel_model, reffe; conformity=:Hcurl,dirichlet_tags=["top_boundary", "bottom_boundary"])
  U = TrialFESpace(V,vec_cov_cf)

  ## interpolation
  uh_interp = interpolate(vec_cov_cf,U)
  _e = ( inv_metric_cf ⋅ uh_interp) - ( inv_metric_cf ⋅ vec_cov_cf)
  e_interp =  sqrt(sum(∫( _e⋅(metric_cf⋅_e)*meas_cf )dΩ_error))


  ## L2 projection
  a(u,v) = ∫( u⋅( inv_metric_cf⋅v)*meas_cf )dΩ
  l(v) = ∫( vec_cov_cf⋅(inv_metric_cf⋅v)*meas_cf )dΩ
  op = AffineFEOperator(a,l,U,V)
  uh_l2proj = solve(ls,op)

  vec_proj_h = covariant_basis_cf ⋅ ( inv_metric_cf ⋅ uh_l2proj)

  _e = ( inv_metric_cf ⋅ uh_l2proj) - ( inv_metric_cf ⋅ vec_cov_cf)
  el2_proj =  sqrt(sum(∫( _e⋅(metric_cf⋅_e)*meas_cf )dΩ_error))

  if return_vtk
    cellfields=["uproj"=> vec_proj_cf,
              "uprojh"=>vec_proj_h,
              "eproj"=>vec_proj_cf-vec_proj_h, ]
    writevtk_with_cell_geomap(latlon_geo_map_func(Ω_atlas),Ω_atlas,dir*"/ambient_model_nref$(lvl)_p$(p_fe)",cellfields=cellfields,
          append=false)
  end

  return  el2_proj,e_interp, false
end

### Since we are in 3D, does not have to be in the tangent space of the 3D sphere
# function uX(forward_map)
#   function f(γαβ)
#     xyz = forward_map(γαβ)
#     # VectorValue(xyz[1],xyz[2],xyz[3])
#     # VectorValue(xyz[2], 0.0, 0.0)
#     VectorValue(0.0,xyz[3],xyz[1]^2)
#   end
# end

# MPI.Init()
# ranks = distribute_with_mpi(LinearIndices((prod(MPI.Comm_size(MPI.COMM_WORLD)),)))

# n_ref_lvls = 3
# ps = [0,1]
# ls = LUSolver()

# Dc = 3
# models = get_3D_octree_refined_models(ranks,n_ref_lvls)

# dir = datadir("InterpolationConvergence")
# (i_am_main(ranks) && !isdir(dir)) && mkdir(dir)

# _dir = dir*"/vector_func_$(Dc)D_Hcurl"
# !isdir(_dir) && mkdir(_dir)
# p_convergence_test(ranks,ps,models,L2_projection_Hcurl,_dir,uX,ls,true)
# plot_convergence_from_saved(_dir,"convergence",["L2 proj","Interp"])
