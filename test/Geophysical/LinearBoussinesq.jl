"""
solve the linear Boussineq equations in 3D in steady form using manufactured solutions
u + f(̂n×u) + ∇ᵧ(φ) - bn̂ = f₁
φ + c² ∇ᵧ⋅u = f₂
b + N² u⋅̂n = f₃
"""

module LinearisedBoussinesqTests

using Gridap
using Gridap.Helpers
using Gridap.Algebra
using GridapDistributed
using GridapGeosciences
using GridapP4est
using Test


THICKNESS = 0.19
RADIUS = 1.0

a_e = 6.37e6/125 # m
d = 5000 #m
Lz = 20e3 #m
R = a_e # m radius
u_0 = 20 #m/s
Ωr = 7.292e-5 #1/s
c = 343 #m/s speed of sound
N = 0.01 #1/s bouyancy frequency
ztop = 10e3 #m
dΘ = 1 #K

LH = a_e # m
LV = ztop/THICKNESS
τ = 1/Ωr # s

_d = d/LV
_Lz = Lz/LV
_R = R/LH
_u_0 = u_0*τ/LH
_Ωr = Ωr*τ
_c = c*τ/LH
_N = N*τ
_ztop = ztop/LV


function p0(xyz)
  x,y,z = xyz
  θϕr   = xyz2θϕr(xyz)
  θ,ϕ,r = θϕr
  sin(ϕ)
end

function b0(xyz)
  x,y,z = xyz
  θϕr   = xyz2θϕr(xyz)
  θ,ϕ,r = θϕr

  θc = 2*π/3
  ϕc = 0.0

  k = sqrt(x^2 + y^2 + z^2) - _R

  r = _R*acos( sin(ϕc)*sin(ϕ) + cos(ϕc)*cos(ϕ)*cos(θ-θc)    )
  s = _d^2/(_d^2 + r^2)
  b = dΘ*s*sin( 2*π*k/_Lz  )
  b
end

function u0(xyz)
  x,y,z = xyz
  θϕr   = xyz2θϕr(xyz)
  θ,ϕ,r = θϕr

  # u = _u_0*cos(ϕ) #
  # v = 0.0#
  u = -_u_0*y/_R
  v = _u_0*x/_R

  VectorValue(u,v,0.0)
end

function omega(xyz)
  x,y,z = xyz
  θϕr   = xyz2θϕr(xyz)
  θ,ϕ,r = θϕr
  2*_Ωr*sin(ϕ)
end



function linear_boussineseq(atlas_model::GridapDistributed.GenericDistributedDiscreteModel{3,3},
  p_fe::Int,dir::String,
  h::Function,vX::Function,f::Function,b::Function,
  ls=LUSolver(),return_vtk=false;
  _i_am_main=true)

  Dc = num_cell_dims(atlas_model)
  lvl = nref(atlas_model)

  _i_am_main && println("nref = $lvl; p_fe = $p_fe; Dc = $Dc")

  degree = 4*(p_fe+1)
  Ω_atlas = Triangulation(atlas_model)
  dΩ = Measure(Ω_atlas,degree)
  dΩ_error = Measure(Ω_atlas,2*degree)

  Q = TestFESpace(Ω_atlas, ReferenceFE(lagrangian,Float64,p_fe); conformity=:L2)
  P = TrialFESpace(Q)

  V = TestFESpace(Ω_atlas, ReferenceFE(raviart_thomas,Float64,p_fe))
  U = TrialFESpace(V)

  W = TestFESpace(Ω_atlas, ReferenceFE(lagrangian,Float64,p_fe))
  B = TrialFESpace(W)

  Y = MultiFieldFESpace([V, Q, W])
  X = MultiFieldFESpace([U, P, B])


  # metric information
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  metric_cf = MetricCellField(Ω_atlas)
  meas_cf = MeasureCellField(Ω_atlas)
  covariant_basis_cf = transpose∘∇(ambient_map_cf)

  h_cf = h∘ambient_map_cf
  u_cf = meas_cf*((pinvJ∘covariant_basis_cf)⋅(vX∘ambient_map_cf))
  u_proj_cf = covariant_basis_cf ⋅(1.0/meas_cf * u_cf  )
  b_cf = b∘ambient_map_cf
  omega_cf = f∘ambient_map_cf

  p_int = interpolate(h_cf,P)
  u_int = interpolate(u_cf,U)
  b_int = interpolate(b_cf,B)

  ## In 3D, we construct ̃k using the area measure
  n3D = CellField(VectorValue(1.0,0.0,0.0),Ω_atlas)
  ff = Operation(sqrt)(n3D ⋅ (InvMetricCellField(Ω_atlas) ⋅ n3D))
  normal_3D_cf = n3D/ff


  coriolis_term((u,p,b),(v,q,r)) = ∫( omega_cf*( normal_3D_cf ×( metric_cf⋅u*(1.0/meas_cf)  ) )⋅(metric_cf⋅v)*(1.0/meas_cf)  )dΩ
  bouyancy_term(b,v) = ∫( b*(normal_3D_cf⋅v)  )dΩ # v ∈ Hdiv, b ∈ L2

  biform_u((u,p,b),(v,q,r)) = ( ∫( (u⋅ (metric_cf⋅v))*(1.0/meas_cf) )dΩ
                              + coriolis_term((u,p,b),(v,q,r))
                              - ∫( p*(∇⋅v) )dΩ
                              - bouyancy_term(b,v)
                              )

  #### Pressure
  biform_p((u,p,b),(v,q,r)) = ∫( (p*q)*meas_cf )dΩ + ∫( _c^2*(q*(∇⋅u)) )dΩ

  #### Bouyancy
  biform_b((u,p,b),(v,q,r)) = ∫( (b*r)*meas_cf )dΩ + _N^2* bouyancy_term(r,u)

  biformX((u,p,b),(v,q,r)) = biform_u((u,p,b),(v,q,r)) + biform_p((u,p,b),(v,q,r)) + biform_b((u,p,b),(v,q,r))

  # Account for the boundary term from IBP in the RHS forcing operator
  Γ = BoundaryTriangulation(atlas_model;tags=["bottom_boundary","top_boundary"])
  dΓ = Measure(Γ,degree)
  nΓ = get_normal_vector(Γ)

  liformX((v,q,r)) = (
    ∫( (u_int⋅ (metric_cf⋅v))*(1.0/meas_cf) )dΩ
  + ∫( gradient(p_int)⋅v )dΩ # assume regularity to IBP
  + coriolis_term((u_int,p_int,b_int),(v,q,r)) # coriolis term
  - bouyancy_term(b_int,v)
  + ∫( (p_int*q)*meas_cf )dΩ + ∫( _c^2*(q*(∇⋅u_int)) )dΩ
  + ∫( (b_int*r)*meas_cf )dΩ + _N^2* bouyancy_term(r,u_int)
  - ∫( (v⋅nΓ)*p_int )dΓ
  )


  #### Multifield problem
  op = AffineFEOperator(biformX,liformX,X,Y)
  A = get_matrix(op)
  b_vec = get_vector(op)
  ns = numerical_setup(symbolic_setup(ls,A),A)
  x = allocate_in_domain(A); fill!(x,0.0)
  solve!(x,ns,b_vec)
  xh = FEFunction(X,x)
  uh,ph,bh = xh


  uh_proj = covariant_basis_cf ⋅ (1.0/meas_cf*uh)

  _e = u_cf - uh
  e_u =  sqrt(sum(∫( _e⋅(metric_cf⋅_e)*(1.0/meas_cf) )dΩ_error))

  _e = h_cf - ph
  e_p = sqrt(sum(∫( (_e*_e)*meas_cf )dΩ_error))

  _e = b_cf - bh
  e_b = sqrt(sum(∫( (_e*_e)*meas_cf )dΩ_error))

  if return_vtk
    panel_cfs = [h_cf, u_proj_cf, b_cf, ph, uh_proj, bh, h_cf-ph, u_proj_cf-uh_proj , b_cf-bh]
    labels = ["p","u_proj", "b", "ph", "uh_proj", "bh", "ep","eu", "eb"]
    cellfields = map((x,y) -> x=>y, labels,panel_cfs)
    writevtk_with_cell_geomap(AmbientMapCellField(Ω_atlas),Ω_atlas,dir*"/ambient_model_nref$(lvl)_p$p_fe",
    cellfields=cellfields,append=false)
  end

  return e_u, e_p, e_b
end


################################################################################
#### Auto convergence test
################################################################################
function main(models::AbstractArray;ps=[1],_i_am_main=true)
  h = p0
  vX = sphere_tangent_vec_component(u0)
  f = omega
  b = b0

  ls = LUSolver()
  dir = @__DIR__
  p_convergence_auto_test(ps,models,linear_boussineseq,dir,h,vX,f,b,ls;_i_am_main=_i_am_main)

end



end #module
