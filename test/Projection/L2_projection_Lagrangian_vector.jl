function L2_projection_Lagrangian_vector(panel_model,
                                         p_fe::Int,
                                         dir::String,
                                         vecX::Function,
                                         conf,
                                         ls=LUSolver(),
                                         return_vtk=true;
                                         _i_am_main=true)

  Dc = num_cell_dims(panel_model)
  lvl = nref(panel_model)

  @check conf in [:L2, :H1] "\n Must be L2 or H1 conformity"

  _i_am_main && println("p_fe = $(p_fe); nref = $lvl; Dc = $Dc, conf = $conf")

  degree = 4*(p_fe+1)

  Ω_panel = Triangulation(panel_model)
  dΩ = Measure(Ω_panel,degree)
  dΩ_error = Measure(Ω_panel,2*degree)

  metric_cf = ParametricCellField(metric,Ω_panel)
  meas_cf = ParametricCellField(sqrtg,Ω_panel)
  covariant_basis_cf = ParametricCellField(covariant_basis,Ω_panel)

  vec_contra_cf = ParametricCellField(contra_v(vecX),Ω_panel)
  vec_proj_cf = covariant_basis_cf⋅vec_contra_cf

  reffe  = ReferenceFE(lagrangian,VectorValue{Dc, Float64},p_fe)
  V = TestFESpace(Ω_panel, reffe; conformity=conf)
  U = TrialFESpace(V)

  if Dc == 3
    V = TestFESpace(Ω_panel, reffe; conformity=conf,
                dirichlet_tags=["top_boundary", "bottom_boundary"])
    U = TrialFESpace(V,vec_contra_cf)
  end

  ## L2 projection
  a(u,v) = ∫( (u⋅(metric_cf⋅v))*meas_cf )dΩ
  l(v) = ∫( (vec_contra_cf⋅(metric_cf⋅v))*meas_cf )dΩ
  op = AffineFEOperator(a,l,U,V)
  vec_contra_h = solve(ls,op)
  vec_l2proj_h = covariant_basis_cf ⋅vec_contra_h

  _e = vec_contra_cf - vec_contra_h
  el2_proj =  sqrt(sum(∫( _e⋅(metric_cf⋅_e)*meas_cf )dΩ_error))

  # Interpolation
  vec_contra_h = interpolate(vec_contra_cf, U)
  vec_interp_h = covariant_basis_cf ⋅vec_contra_h
  _e = vec_contra_cf - vec_contra_h
  el2_interp =  sqrt(sum(∫( _e⋅(metric_cf⋅_e)*meas_cf )dΩ_error))

  # _i_am_main && println("Error interp: ", el2_interp)
  # _i_am_main && println("Error proj: ", el2_proj)

  if return_vtk
    panel_cfs = [vec_proj_cf, vec_l2proj_h, vec_proj_cf-vec_l2proj_h,
                vec_interp_h, vec_interp_h-vec_proj_cf]
    labels = ["u_proj", "u_projh", "eproj",
              "u_int", "e_int"]

    cellfields = map((x,y) -> x=>y, labels,panel_cfs)
    writevtk_with_cell_geomap(geo_map_func(Ω_panel),Ω_panel,dir*"/ambient_model_nref$(lvl)_p$(p_fe)",cellfields=cellfields,
          append=false)
  end

  return  el2_proj,el2_interp,false

end

### must be in the tangent space of the sphere
# function uX(p)
#   function _u(α)
#     x = CubedSphereForwardMap(p)(α)
#     VectorValue(-x[2],x[1],0.0)
#   end
# end

# MPI.Init()
# ranks = distribute_with_mpi(LinearIndices((prod(MPI.Comm_size(MPI.COMM_WORLD)),)))

# n_ref_lvls = 4
# ps = [1,2]
# ls = LUSolver()

# dir = datadir("InterpolationConvergence")
# !isdir(dir) && mkdir(dir)

# Dc = 3
# models = (Dc == 2) ? get_octree_refined_models(ranks,n_ref_lvls,radius) : get_3D_octree_refined_models(ranks,n_ref_lvls-1)

# Dc = 2
# models = get_refined_models(n_ref_lvls)
# for conf in [:L2, :H1]
#   _dir = dir*"/vector_func_$(Dc)D_"*String(conf)
#   !isdir(_dir) && mkdir(_dir)
#   p_convergence_test(ranks,ps,models,L2_projection_Lagrangian_vector,_dir,uX,conf,ls,true)
#   plot_convergence_from_saved(_dir,"convergence",["L2Proj","Interp" ])
# end
