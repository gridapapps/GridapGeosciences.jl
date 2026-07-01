function L2_projection_Lagrangian_vector(atlas_model,
                                         p_fe::Int,
                                         dir::String,
                                         vecX::Function,
                                         conf,
                                         ls=LUSolver(),
                                         return_vtk=true;
                                         _i_am_main=true)

  Dc = num_cell_dims(atlas_model)
  lvl = nref(atlas_model)

  @check conf in [:L2, :H1] "\n Must be L2 or H1 conformity"

  _i_am_main && println("L2_projection_Lagrangian_vector: p_fe = $(p_fe); nref = $lvl; Dc = $Dc, conf = $conf")

  degree = 4*(p_fe+1)

  Ω_atlas = Triangulation(atlas_model)
  dΩ = Measure(Ω_atlas,degree)
  dΩ_error = Measure(Ω_atlas,2*degree)

  metric_cf = MetricCellField(Ω_atlas)
  meas_cf = MeasureCellField(Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  covariant_basis_cf = transpose∘∇(ambient_map_cf)
  vec_contra_cf = (pinvJ∘covariant_basis_cf)⋅(vecX∘ambient_map_cf)
  vec_proj_cf = covariant_basis_cf⋅vec_contra_cf

  reffe  = ReferenceFE(lagrangian,VectorValue{Dc, Float64},p_fe)
  V = TestFESpace(Ω_atlas, reffe; conformity=conf)
  U = TrialFESpace(V)

  if Dc == 3
    V = TestFESpace(Ω_atlas, reffe; conformity=conf,
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
    writevtk_with_cell_geomap(AmbientMapCellField(Ω_atlas),Ω_atlas,dir*"/ambient_model_nref$(lvl)_p$(p_fe)",cellfields=cellfields,
          append=false)
  end

  return  el2_proj,el2_interp,false

end
