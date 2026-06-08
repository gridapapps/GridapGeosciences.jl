"""
solve the linearised shallow water equations in steady form using manufactured solutions
u + f (k×u) + ∇ᵧ(φ) = f₁
φ + ∇ᵧ⋅u = f₁
"""

module LinearisedShallowWaterTests

using Gridap
using Gridap.Helpers
using Gridap.Algebra
using GridapGeosciences
using GridapP4est
using Test

include("Williamson_functions.jl")


a_e = 6.37e6 # m
g = 9.8 # m/2
ω = 7.29e-5 #s^-1
T = 12*24*3600 #s
H_0 = 2.94e4/g #m
u_0 = 2*π*a_e/T #m/s

L = a_e
_τ = 1/ω

_a = a_e/L
_g = g*_τ^2/L
_ω = ω*_τ
_H_0 = H_0/L
_T = T/_τ
_u0 = u_0/L*_τ


function linear_shallow_water_solver(atlas_model,
  p_fe::Int,dir::String,h::Function,vX::Function,f::Function,ls=LUSolver(),return_vtk=false;
  _i_am_main=true)

  Dc = num_cell_dims(atlas_model)
  lvl = nref(atlas_model)

  _i_am_main && println("nref = $lvl; p_fe = $p_fe; Dc = $Dc")

  degree = 5*(p_fe+1)
  Ω_atlas = Triangulation(atlas_model)
  dΩ = Measure(Ω_atlas,degree)
  dΩ_error = Measure(Ω_atlas,2*degree)

  Q = TestFESpace(Ω_atlas, ReferenceFE(lagrangian,Float64,p_fe); conformity=:L2)
  P = TrialFESpace(Q)

  V = TestFESpace(Ω_atlas, ReferenceFE(raviart_thomas,Float64,p_fe); conformity=:HDiv)
  U = TrialFESpace(V)

  Y = MultiFieldFESpace([V, Q])
  X = MultiFieldFESpace([U, P])

  # metric information
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  metric_cf = MetricCellField(Ω_atlas)
  inv_metric_cf = InvMetricCellField(Ω_atlas)
  meas_cf = sqrt∘det∘metric_cf
  covariant_basis_cf = transpose∘∇(ambient_map_cf)

  h_cf = h∘ambient_map_cf
  u_cf = meas_cf*((pinvJ∘covariant_basis_cf)⋅(vX∘ambient_map_cf))
  u_proj_cf = covariant_basis_cf ⋅(1.0/meas_cf * u_cf  )
  cor_cf = f∘ambient_map_cf

  p_int = interpolate(h_cf,P)
  u_int = interpolate(u_cf,U)

  ## Here we construct the coriolis term in 2D v. 3D
  ## On the surface, the term is: ∫( ̃f ( ̃k × ̃u  )  )dΩ
  ##
  ## In 2D, we use the rotation matrx
  function vecPerp(u)
    # u   = (u1, u2)
    # u^T = (-u2, u1)
    VectorValue(-u[2],u[1])
  end
  Aperp = [0 -1
          1 0]
  Rperp = TensorValue(Aperp)
  Rperp_cf = CellField(Rperp,Ω_atlas)

  ## In 3D, we construct ̃k using the area measure
  #_area_meas(p) = x->  forward_jacobian(p,x) ⋅ (inv_metric(p,x) ⋅ VectorValue(1,0,0))
  # area_meas(p) = x-> norm(_area_meas(p)(x))
  # normal_3D(p) = x-> (1/area_meas(p)(x) )*VectorValue(1,0,0)
  normal_3D_cf = 1.0/(norm∘(covariant_basis_cf⋅(inv_metric_cf⋅VectorValue(1.0,0.0,0.0))))*VectorValue(1.0,0.0,0.0) 

  ## return the appropriate term based on Dimension
  function get_coriolis_term(Dc::Int)
    if Dc == 2
      return ((u,p),(v,q)) -> ∫( ( cor_cf*( (Rperp_cf⋅ u)⋅v))  )dΩ
    elseif Dc == 3
      return ((u,p),(v,q)) -> ∫( cor_cf*( normal_3D_cf ×( metric_cf⋅u*(1.0/meas_cf)  ) )⋅(metric_cf⋅v)*(1.0/meas_cf)  )dΩ
    end
  end
  coriolis_term((u,p),(v,q)) = get_coriolis_term(Dc)((u,p),(v,q))

  ## construct bilinear form using coriolis_term
  biform_u((u,p),(v,q)) = ( ∫( (u⋅ (metric_cf⋅v))*(1.0/meas_cf) )dΩ
                        + coriolis_term((u,p),(v,q))
                         - ∫( p*(∇⋅v) )dΩ
                          )
  biform_p((u,p),(v,q)) = ∫( (p*q)*meas_cf )dΩ + ∫( q*(∇⋅u) )dΩ
  biformX((u,p),(v,q)) = biform_u((u,p),(v,q)) + biform_p((u,p),(v,q))


  # manufacture rhs functions
  function get_liform(Dc::Int)

    # the manufactured solution is exactly the LHS operator
    _liformX((v,q)) = (
      ∫( (u_int⋅ (metric_cf⋅v))*(1.0/meas_cf) )dΩ
    + ∫( gradient(p_int)⋅v )dΩ # assume regularity to IBP
    + coriolis_term((u_int,p_int),(v,q)) # coriolis term
    + ∫( (p_int*q)*meas_cf )dΩ
    + ∫( q*(∇⋅u_int) )dΩ
    )

    if Dc == 2
      return v -> _liformX(v)
    elseif Dc == 3
      # in 3D, account for the boundary term from IBP
      Γ = BoundaryTriangulation(atlas_model;tags=["bottom_boundary","top_boundary"])
      dΓ = Measure(Γ,degree)
      nΓ = get_normal_vector(Γ)
      boundary((v,q)) = ∫( (v⋅nΓ)*p_int )dΓ
      return v -> _liformX(v) - boundary(v)
    end
  end


  op = AffineFEOperator(biformX,get_liform(Dc),X,Y)
  uh,ph = solve(ls,op)

  uh_proj = covariant_basis_cf ⋅ (1.0/meas_cf*uh)

  _e = u_cf - uh
  e_u =  sqrt(sum(∫( _e⋅(metric_cf⋅_e)*(1.0/meas_cf) )dΩ_error))

  _e = h_cf - ph
  e_p = sqrt(sum(∫( (_e*_e)*meas_cf )dΩ_error))

  if return_vtk
    panel_cfs = [ph, uh_proj, uh_proj-u_proj_cf,ph-h_cf]
    labels = ["p","u_proj","eu","ep"]
    cellfields = map((x,y) -> x=>y, labels,panel_cfs)
    writevtk_with_cell_geomap(geo_map_func(Ω_atlas),Ω_atlas,dir*"/ambient_model_nref$(lvl)_p$(p_fe)_D$Dc",
          cellfields=cellfields,append=false)
  end

  return e_u, e_p, false

end


################################################################################
#### Auto convergence test
################################################################################
function main(models::AbstractArray;ps=[2],_i_am_main=true)
  h = h₀(0.0)
  vX = tangent_vec(u₀(0.0))
  f = f₀(0.0)

  ls = LUSolver()
  dir = @__DIR__
  p_convergence_auto_test(ps,models,linear_shallow_water_solver,dir,h,vX,f,ls;_i_am_main=_i_am_main)
end




end # module
