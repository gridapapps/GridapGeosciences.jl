module AmbientLinearisedShallowWaterTests

using Gridap
using Gridap.Helpers
using Gridap.Algebra
using GridapGeosciences
using GridapP4est
using Test

include("../Geophysical/Williamson_functions.jl")


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

function linear_shallow_water_solver(
  extrinsic_atlas_model,
  p_fe::Int,dir::String,h::Function,vX::Function,f::Function,ls=LUSolver(),return_vtk=false;
  _i_am_main=true)

  Dc = num_cell_dims(extrinsic_atlas_model)
  lvl = nref(extrinsic_atlas_model)

  _i_am_main && println("nref = $lvl; p_fe = $p_fe; Dc = $Dc")

  degree = 5*(p_fe+1)
  Ω_ambient = Triangulation(extrinsic_atlas_model)
  dΩ = Measure(Ω_ambient,degree)
  dΩ_error = Measure(Ω_ambient,2*degree)

  Q = TestFESpace(Ω_ambient, ReferenceFE(lagrangian,Float64,p_fe); conformity=:L2)
  P = TrialFESpace(Q)

  V = TestFESpace(Ω_ambient, ReferenceFE(raviart_thomas,Float64,p_fe); conformity=:HDiv)
  U = TrialFESpace(V)

  Y = MultiFieldFESpace([V, Q])
  X = MultiFieldFESpace([U, P])

  p_int = interpolate(h,P)
  u_int = interpolate(vX,U)

  ## Here we construct the coriolis term on the surface: ∫( ̃f ( ̃k × ̃u  )  )dΩ
  n_surf = get_surface_normal(Ω_ambient)
  coriolis_term((u,p),(v,q)) = ∫( f*( ( n_surf × u)⋅v)  )dΩ

  ## construct bilinear form using coriolis_term
  biform_u((u,p),(v,q)) = ( ∫( u⋅v )dΩ
                        + coriolis_term((u,p),(v,q))
                         - ∫( p*(∇⋅v) )dΩ
                          )
  biform_p((u,p),(v,q)) = ∫( p*q )dΩ + ∫( q*(∇⋅u) )dΩ
  biformX((u,p),(v,q)) = biform_u((u,p),(v,q)) + biform_p((u,p),(v,q))


  # manufacture rhs functions
  function get_liform(Dc::Int)

    # the manufactured solution is exactly the LHS operator
    _liformX((v,q)) = (
      ∫( u_int⋅v )dΩ
    + ∫( gradient(p_int)⋅v )dΩ # assume regularity to IBP
    + coriolis_term((u_int,p_int),(v,q)) # coriolis term
    + ∫( p_int*q )dΩ
    + ∫( q*(∇⋅u_int) )dΩ
    )

    if Dc == 2
      return v -> _liformX(v)
    elseif Dc == 3
      # in 3D, account for the boundary term from IBP
      Γ = BoundaryTriangulation(extrinsic_atlas_model;tags=["bottom_boundary","top_boundary"])
      dΓ = Measure(Γ,degree)
      nΓ = get_normal_vector(Γ)
      boundary((v,q)) = ∫( (v⋅nΓ)*p_int )dΓ
      return v -> _liformX(v) - boundary(v)
    end
  end


  op = AffineFEOperator(biformX,get_liform(Dc),X,Y)
  uh,ph = solve(ls,op)

  _e = vX - uh
  e_u =  sqrt(sum(∫( _e⋅_e )dΩ_error))

  _e = h - ph
  e_p = sqrt(sum(∫( _e*_e )dΩ_error))

  if return_vtk
    panel_cfs = [ph, uh, uh-vX,ph-h]
    labels = ["p","u_proj","eu","ep"]
    cellfields = map((x,y) -> x=>y, labels,panel_cfs)
    writevtk(Ω_ambient,dir*"/ambient_model_nref$(lvl)_p$(p_fe)_D$Dc",
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
