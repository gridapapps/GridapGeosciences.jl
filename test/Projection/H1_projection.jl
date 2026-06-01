using MPI
using PartitionedArrays
using GridapGeosciences
using GridapGeosciences.Distributed
using GridapP4est
using Gridap
using GridapDistributed



### f = sin(ϕ) = Z/R
function func(p)
  function _f(αβ)
    X,Y,Z = CubedSphereForwardMap(p)(αβ)
    radius = sqrt(X^2 + Y^2 + Z^2)
    Z/radius
    # X^2
  end
end


MPI.Init()
ranks = distribute_with_mpi(LinearIndices((prod(MPI.Comm_size(MPI.COMM_WORLD)),)))



function interpolation(panel_model,p_fe::Int,dir::String,func::Function,return_vtk)


  Dc = num_cell_dims(panel_model)

  lvl = nref(nc(panel_model))


  println("p_fe = $(p_fe); nref = $lvl; Dc = $Dc")

  Ω_panel = Triangulation(panel_model)
  dΩ = Measure(Ω_panel,10*p_fe)
  panel_ids = get_panel_ids(panel_model)

  f_panel_cf = ParametricCellField(func,Ω_panel,panel_ids)
  meas_cf = ParametricCellField(sqrtg,Ω_panel,panel_ids)
  inv_metric_cf = ParametricCellField(inv_metric,Ω_panel,panel_ids)

  slap_panel_cf =  ParametricCellField(surflap(func),Ω_panel,panel_ids)

  i_am_main(ranks) && println("Zeromean: ", sum(∫(f_panel_cf*meas_cf)dΩ))


  rhs_cf = f_panel_cf - slap_panel_cf

  V = TestFESpace(panel_model, ReferenceFE(lagrangian,Float64,p_fe); conformity=:H1)
  U = TrialFESpace(V)
  a(u,v) = ∫( (u*v)*meas_cf )dΩ + ∫( ( ∇(v)⋅(inv_metric_cf⋅∇(u)))*meas_cf )dΩ

  l(v) = ∫( (f_panel_cf*v)*meas_cf )dΩ + ∫( ( ∇(v)⋅(inv_metric_cf⋅∇(f_panel_cf)))*meas_cf )dΩ
  # l(v) =   ∫(  (rhs_cf*v)*meas_cf )dΩ


  op = AffineFEOperator(a,l,U,V)
  f_uh = solve(LUSolver(),op)


  _e = f_panel_cf - f_uh
  el2  =  sqrt( sum(∫( (_e*_e)*meas_cf )dΩ) )
  eh1  =  sqrt( sum( ∫( (_e*_e)*meas_cf )dΩ + ∫( (∇(_e)⋅ (inv_metric_cf⋅∇(_e)) )*meas_cf )dΩ   )  )

  # if return_vtk
    panel_cfs = [f_panel_cf, f_uh,  _e, gradient(f_uh) ]
    labels = ["u","uh", "e" , "grad"]
    cellfields = map((x,y) -> x=>y, labels,panel_cfs)
    writevtk_with_cell_geomap(latlon_geo_map_func(Ω_panel),Ω_panel,dir*"/ambient_model_nref$(lvl)_p$(p_fe)_"*String(:H1),
            cellfields=cellfields,append=false)
  # end

  return el2,eh1,false
end

n_ref_lvls = 4
ps = [1]

dir = datadir("InterpolationConvergence")
!isdir(dir) && mkdir(dir)

models = get_octree_refined_models(ranks,n_ref_lvls,radius)
_dir = dir*"/H1_interpolate_scalar_func_2D"
!isdir(_dir) && mkdir(_dir)
p_convergence_test(ranks,ps,models,interpolation,_dir,func,true)
plot_convergence_from_saved(_dir,"convergence",["l2", "h1"])
