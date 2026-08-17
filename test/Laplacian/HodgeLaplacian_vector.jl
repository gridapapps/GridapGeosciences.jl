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

function uX(xyz)
  r = sqrt(xyz[1]^2 + xyz[2]^2 + xyz[3]^2)
  f = 2.0*xyz[3]/r
  n = sphere_surface_normal_vec(xyz)
  f*n
end

function hodge_laplacian_vector(
  atlas_model::Union{<:IntrinsicAtlasDiscreteModel{3,3},
                     <:Gridap.Adaptivity.AdaptedDiscreteModel{3,3,<:IntrinsicAtlasDiscreteModel{3,3}},
                     <:GridapGeosciences.IntrinsicAtlasDistributedDiscreteModel{3,3},
                     <:GridapGeosciences.AdaptedIntrinsicAtlasDistributedDiscreteModel{3,3},
                     <:GridapGeosciences.AtlasOctreeDistributedDiscreteModel{3,3,<:Any,<:Any,<:IntrinsicManifold}},
  p_fe::Int,dir::String,uX::Function,ls=LUSolver(),return_vtk=false;
  _i_am_main=true)

  Dc = num_cell_dims(atlas_model)
  lvl = nref(atlas_model)
 _i_am_main && println("p_fe = $(p_fe); nref = $lvl; Dc = $Dc")

  # degree = 30
  degree = 5*(p_fe + 1)
  if p_fe == 0
    degree = 10
  end
  @check degree > 0 "Zero quad!!"

  ## finite element solver
  Ω_atlas = Triangulation(atlas_model)
  dΩ = Measure(Ω_atlas,degree)
  Ω_error = Triangulation(atlas_model)
  dΩ_error = Measure(Ω_error,2*degree)

  tags = ["top_boundary", "bottom_boundary"]
  Γ = BoundaryTriangulation(atlas_model,tags=tags)
  dΓ = Measure(Γ,degree)
  nΓ = get_normal_vector(Γ)

  ## metric information
  inv_metric_cf = InvMetricCellField(Ω_atlas)
  metric_cf = MetricCellField(Ω_atlas)
  meas_cf = MeasureCellField(Ω_atlas)

  ## ambient map and jacobian
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  Jt_cf = ∇(ambient_map_cf)
  covariant_basis_cf = transpose∘Jt_cf

  u_cf = uX ∘ ambient_map_cf

  ### Rhs function (covariant vector of the surface vector Laplacian of uX)
  rhs_cov_cf = -vecΔs(uX, Ω_atlas)

  u_cov_cf = Jt_cf⋅u_cf

  ### Covariant vector of the surface curl of uX
  curls_u_cf = curls(uX, Ω_atlas)

  sdiv_cf =  divs(uX, Ω_atlas)
  sigma_cf = -sdiv_cf

  # cellfields = ["curlu"=>ccurlu_cov_cf,
  #               "u"=>covariant_basis_cf ⋅ (inv_metric_cf⋅u_cov_cf),
  #               "un"=>un_cf,
  #               "curlu_cross"=>covariant_basis_cf ⋅ (inv_metric_cf⋅curlu_cross),
  #               "sigma"=>-sdiv_cf,
  #               "rhs"=>rhs_cov_cf
  #               ]
  # writevtk_with_cell_geomap(AmbientMapCellField(Ω_atlas),Ω_atlas,dir*"/sol",
  #         cellfields=cellfields,
  #         append=false)


  ## FE spaces
  T = TestFESpace(Ω_atlas, ReferenceFE(lagrangian,Float64,p_fe+1); conformity=:H1)
  S = TrialFESpace(T)

  R = TestFESpace(Ω_atlas, ReferenceFE(nedelec,Float64,p_fe); conformity=:Hcurl)
  H = TrialFESpace(R)

  sigma_int = interpolate(sigma_cf,S)
  u_int = interpolate(u_cov_cf,H)

  ### Multifield
  X = MultiFieldFESpace([S,H])
  Y = MultiFieldFESpace([T,R])

  biform_x((s,u),(t,v)) = (
                  ∫( (s*t)*meas_cf  )dΩ
                - ∫( ∇(t)⋅(inv_metric_cf⋅u)*meas_cf  )dΩ
                + ∫( curl(u)⋅(metric_cf⋅curl(v))*(1.0/meas_cf) )dΩ
                + ∫( gradient(s)⋅(inv_metric_cf⋅v)*meas_cf )dΩ
                  )
  liform_x((t,v)) = (
                ∫( rhs_cov_cf⋅(inv_metric_cf⋅v)*meas_cf  )dΩ
                + ∫( v⋅( ( curls_u_cf )×nΓ)     )dΓ
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
    writevtk_with_cell_geomap(AmbientMapCellField(Ω_atlas),Ω_atlas,dir*"/ambient_model_nref$(lvl)_p$p_fe",
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
