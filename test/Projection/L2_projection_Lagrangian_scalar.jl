
function L2_projection_Lagrangian_scalar(panel_model,
                                         p_fe::Int,
                                         dir::String,
                                         func::Function,
                                         conf,
                                         ls=LUSolver(),
                                         return_vtk=false;
                                         _i_am_main=true)

  Dc = num_cell_dims(panel_model)
  lvl = nref(panel_model)

  @check conf in [:L2, :H1] "\n Must be L2 or H1 conformity"

  _i_am_main && println("p_fe = $(p_fe); nref = $lvl; Dc = $Dc, conf = $conf")

  degree = 4*(p_fe+1)

  Ω_panel = Triangulation(panel_model)
  dΩ = Measure(Ω_panel,degree)
  dΩ_error = Measure(Ω_panel,2*degree)

  f_cf = ParametricCellField(func,Ω_panel)
  meas_cf = ParametricCellField(sqrtg,Ω_panel)

  V = TestFESpace(panel_model, ReferenceFE(lagrangian,Float64,p_fe); conformity=conf)
  U = TrialFESpace(V)

  if Dc == 3
    V = TestFESpace(panel_model, ReferenceFE(lagrangian,Float64,p_fe); conformity=conf,
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
    writevtk_with_cell_geomap(latlon_geo_map_func(Ω_panel),Ω_panel,dir*"/ambient_model_nref$(lvl)_p$(p_fe)_"*String(conf),
            cellfields=cellfields,append=false)
  end

  return e_l2proj,e_interp,false

end

# function fS(p)
#   function f(αβ)
#     xyz = CubedSphereForwardMap(p)(αβ)
#     xyz[1]*xyz[2]*xyz[3]
#   end
# end

# ### f = sin(ϕ) = Z/R
# function fS(p)
#   function _f(αβ)
#     X,Y,Z = CubedSphereForwardMap(p)(αβ)
#     radius = sqrt(X^2 + Y^2 + Z^2)
#     Z/radius
#   end
# end


# MPI.Init()
# ranks = distribute_with_mpi(LinearIndices((prod(MPI.Comm_size(MPI.COMM_WORLD)),)))


# n_ref_lvls = 4
# ps = [1,2]
# ls = LUSolver()

# dir = datadir("InterpolationConvergence")
# !isdir(dir) && mkdir(dir)

# Dc = 2

# # models = (Dc == 2) ? get_octree_refined_models(ranks,n_ref_lvls,radius) : get_3D_octree_refined_models(ranks,n_ref_lvls)
# models = get_refined_models(n_ref_lvls)
# for conf in [:L2, :H1]
#   _dir = dir*"/scalar_func_$(Dc)D_"*String(conf)
#   !isdir(_dir) && mkdir(_dir)
#   p_convergence_test(ranks,ps,models,L2_projection_Lagrangian_scalar,_dir,fS,conf,ls,true)
#   plot_convergence_from_saved(_dir,"convergence",["L2Proj","Interp" ])
# end
