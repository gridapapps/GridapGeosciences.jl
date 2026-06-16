
function L2_projection_Lagrangian_scalar(atlas_model,
                                         p_fe::Int,
                                         dir::String,
                                         func::Function,
                                         conf,
                                         ls=LUSolver(),
                                         return_vtk=false;
                                         _i_am_main=true)

  Dc = num_cell_dims(atlas_model)
  lvl = nref(atlas_model)

  @check conf in [:L2, :H1] "\n Must be L2 or H1 conformity"

  _i_am_main && println("L2_projection_Lagrangian_scalar: p_fe = $(p_fe); nref = $lvl; Dc = $Dc, conf = $conf")

  degree = 4*(p_fe+1)

  Ω_atlas = Triangulation(atlas_model)
  dΩ = Measure(Ω_atlas,degree)
  dΩ_error = Measure(Ω_atlas,2*degree)

  f_cf = func∘AmbientMapCellField(Ω_atlas)
  meas_cf = sqrt∘det∘MetricCellField(Ω_atlas)

  V = TestFESpace(atlas_model, ReferenceFE(lagrangian,Float64,p_fe); conformity=conf)
  U = TrialFESpace(V)

  if Dc == 3
    V = TestFESpace(atlas_model, ReferenceFE(lagrangian,Float64,p_fe); conformity=conf,
                    dirichlet_tags=["top_boundary", "bottom_boundary"])
    U = TrialFESpace(V,f_cf)
  end

  ## interpolation
  fh_interp = interpolate(f_cf,U)
  _e = f_cf - fh_interp
  e_interp  =  sqrt( sum(∫( (_e*_e)*meas_cf )dΩ_error) )

  ## L2 projection
  a(u,v) = ∫( (u*v)*meas_cf )dΩ
  l(v) = ∫( (f_cf*v)*meas_cf )dΩ
  op = AffineFEOperator(a,l,U,V)
  fh_l2proj = solve(ls,op)

  _e = f_cf - fh_l2proj
  e_l2proj  =  sqrt( sum(∫( (_e*_e)*meas_cf )dΩ_error) )

  if return_vtk
    panel_cfs = [f_cf, fh_l2proj,  _e, gradient(fh_l2proj) ]
    labels = ["u","uh", "e" , "grad"]
    cellfields = map((x,y) -> x=>y, labels,panel_cfs)
    writevtk_with_cell_geomap(latlon_geo_map_func(Ω_atlas),Ω_atlas,dir*"/ambient_model_nref$(lvl)_p$(p_fe)_"*String(conf),
            cellfields=cellfields,append=false)
  end

  return e_l2proj,e_interp,false

end