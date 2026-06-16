"""
solve the linearised wave equation in steady form using manufactured solutions
u + ∇ᵧ(φ) = f₁
φ + ∇ᵧ⋅u = f₁
"""

module WaveEquationTests

using Gridap
using Gridap.Helpers
using Gridap.Algebra
using GridapGeosciences
using GridapP4est
using Test

include("Williamson_functions.jl")

function topography(xyz)
  0.0
end

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




function wave_solver(atlas_model,
  p_fe::Int,dir::String,h::Function,vX::Function,ls=LUSolver(),return_vtk=false;
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
  meas_cf = MeasureCellField(Ω_atlas)
  covariant_basis_cf = transpose∘∇(ambient_map_cf)

  h_cf = h∘ambient_map_cf
  u_cf = meas_cf*((pinvJ∘covariant_basis_cf)⋅(vX∘ambient_map_cf))
  u_proj_cf = covariant_basis_cf ⋅(1.0/meas_cf*u_cf)

  p_int = interpolate(h_cf,P)
  u_int = interpolate(u_cf,U)

  biform1((u,p),(v,q)) = ∫( (u⋅ (metric_cf⋅v))*(1.0/meas_cf) )dΩ - ∫( p*(∇⋅v) )dΩ
  biform2((u,p),(v,q)) = ∫( (p*q)*meas_cf )dΩ + ∫( q*(∇⋅u) )dΩ
  biformX((u,p),(v,q)) = biform1((u,p),(v,q)) + biform2((u,p),(v,q))

  # manufacture rhs functions
  function get_liform(Dc::Int)

    # the manufactured solution is exactly the LHS operator
    _liformX((v,q)) = (
      ∫( (u_int⋅ (metric_cf⋅v))*(1.0/meas_cf) )dΩ
    + ∫( gradient(p_int)⋅v )dΩ # assume regularity to IBP
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
    panel_cfs = [h_cf, u_proj_cf, ph, uh_proj, h_cf-ph, u_proj_cf-uh_proj ]
    labels = ["p","u_proj", "ph", "uh_proj", "ep","eu"]

    cellfields = map((x,y) -> x=>y, labels,panel_cfs)
    writevtk_with_cell_geomap(AmbientMapCellField(Ω_atlas),Ω_atlas,dir*"/ambient_model_nref$(lvl)_p$(p_fe)_D$Dc",
            cellfields=cellfields,append=false)
  end

  return e_u,e_p,false
end


################################################################################
#### Auto convergence test
################################################################################
function main(models::AbstractArray;ps=[2],_i_am_main=true)
  h = h₀(0.0)
  vX = tangent_vec(u₀(0.0))

  ls = LUSolver()
  dir = @__DIR__
  p_convergence_auto_test(ps,models,wave_solver,dir,h,vX,ls;_i_am_main=_i_am_main)
end



end # module
