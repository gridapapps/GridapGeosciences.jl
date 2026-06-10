function L2_projection_Hdiv(atlas_model,
                            p_fe::Int,
                            dir::String,
                            vecX::Function,
                            ls=LUSolver(),
                            return_vtk=false;
                            _i_am_main=true)

  Dc = num_cell_dims(atlas_model)
  lvl = nref(atlas_model)

  _i_am_main && println("p_fe = $(p_fe); nref = $lvl; Dc = $Dc")

  degree = 4*(p_fe+1)
  Ω_atlas = Triangulation(atlas_model)
  dΩ = Measure(Ω_atlas,degree)
  dΩ_error = Measure(Ω_atlas,2*degree)

  metric_cf = MetricCellField(Ω_atlas)
  meas_cf = MeasureCellField(Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  covariant_basis_cf = transpose∘∇(ambient_map_cf)

  ### Piola mapping for Hdiv fields
  vec_piola_cf = meas_cf*((pinvJ∘covariant_basis_cf)⋅(vecX∘ambient_map_cf))
  vec_proj_cf_piola = covariant_basis_cf⋅ ( 1.0/meas_cf* vec_piola_cf )

  reffe = ReferenceFE(raviart_thomas,Float64,p_fe)
  V = TestFESpace(atlas_model, reffe; conformity=:HDiv)
  U = TrialFESpace(V)

  if Dc == 3
    V = TestFESpace(atlas_model, reffe; conformity=:Hdiv,dirichlet_tags=["top_boundary", "bottom_boundary"])
    U = TrialFESpace(V,vec_piola_cf)
  end

  ## Interpolation
  uh_interp = interpolate(vec_piola_cf,U)
  _e = vec_piola_cf - uh_interp
  e_interp =  sqrt(sum(∫( _e⋅(metric_cf⋅_e)*(1.0/meas_cf) )dΩ_error))

  ## L2 projection
  a(u,v) = ∫( u⋅( metric_cf⋅v)*(1.0/meas_cf) )dΩ
  l(v) = ∫( vec_piola_cf⋅(metric_cf⋅v)*(1.0/meas_cf) )dΩ
  op = AffineFEOperator(a,l,U,V)
  uh_l2proj = solve(ls,op)

  vec_proj_h = covariant_basis_cf ⋅((1.0/meas_cf) * uh_l2proj)
  _e = vec_piola_cf - uh_l2proj
  e_l2proj =  sqrt(sum(∫( _e⋅(metric_cf⋅_e)*(1.0/meas_cf) )dΩ_error))

  if return_vtk
    cellfields = [
              "u_cf"=>vec_proj_cf_piola,
              "uh"=>vec_proj_h,
              "eu"=>vec_proj_cf_piola-vec_proj_h  ]

    writevtk_with_cell_geomap(latlon_geo_map_func(Ω_atlas),Ω_atlas,dir*"/ambient_model_nref$(lvl)_p$(p_fe)",cellfields=cellfields,
          append=false)
  end

  return e_l2proj,e_interp,false
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
# ps = [1]
# ls = LUSolver()

# Dc = 3
# models = (Dc == 2) ? get_octree_refined_models(ranks,n_ref_lvls,radius) : get_3D_octree_refined_models(ranks,n_ref_lvls-1)

# dir = datadir("InterpolateConvergence")
# (i_am_main(ranks) && !isdir(dir)) && mkdir(dir)

# _dir = dir*"/vector_func_$(Dc)D_Hdiv"
# !isdir(_dir) && mkdir(_dir)

# p_convergence_test(ranks,ps,models,L2_projection_Hdiv,_dir,uX,ls,true)
# plot_convergence_from_saved(_dir,"convergence",["L2 proj","Interp"])
