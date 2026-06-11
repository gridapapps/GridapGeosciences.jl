"""
solve scalar laplacian in mixed form
u + ∇ᵧ(φ) = 0
∇ᵧ⋅u = f
where f = -Δφ
"""

module AmbientHodgeLaplacianScalarTests

using Gridap
using Gridap.Helpers
using Gridap.Algebra
using GridapGeosciences
using GridapP4est
using Test


function fX(xyz)
    θϕr   = xyz2θϕr(xyz)
    sin(θϕr[2])
end


function hodge_laplacian_scalar(
  extrinsic_ambient_model,
  p_fe::Int,dir::String,f::Function,ls=LUSolver(),return_vtk=false;
  _i_am_main=true)

  Dc = num_cell_dims(extrinsic_ambient_model)
  lvl = nref(extrinsic_ambient_model)
 _i_am_main && println("p_fe = $(p_fe); nref = $lvl; Dc = $Dc")

  degree = 4*(p_fe+1)
  if p_fe == 0
    degree = 10
  end
  @check degree > 0 "Zero quad!!"

  Ω_ambient = Triangulation(extrinsic_ambient_model)
  dΩ = Measure(Ω_ambient,degree)
  dΩ_error = Measure(Ω_ambient,2*degree)

  # FE spaces
  Q = TestFESpace(Ω_ambient, ReferenceFE(lagrangian,Float64,p_fe); conformity=:L2)
  if Dc == 2
   _i_am_main && println("zeromean constraint in 2D ")
    Q = TestFESpace(Ω_ambient, ReferenceFE(lagrangian,Float64,p_fe); conformity=:L2, constraint=:zeromean)
  end
  P = TrialFESpace(Q)

  V = TestFESpace(Ω_ambient, ReferenceFE(raviart_thomas,Float64,p_fe); conformity=:HDiv)
  U = TrialFESpace(V)

  Y = MultiFieldFESpace([V, Q])
  X = MultiFieldFESpace([U, P])

  # manufactured RHS
  sigma_cf = ∇s(f,Ω_ambient)
  slap_cf = Δs(f,Ω_ambient)
  rhs_cf = -slap_cf

  f_int = interpolate(f,P)

  biform_u((u,p),(v,q)) = ∫( u⋅v )dΩ - ∫( p*(∇⋅v) )dΩ
  biform_p((u,p),(v,q)) = ∫( q*(∇⋅u) )dΩ

  biformX((u,p),(v,q)) = biform_u((u,p),(v,q)) + biform_p((u,p),(v,q))


  # manufacture rhs functions
  function get_liform(Dc::Int)

    # the manufactured solution
    _liformX((v,q)) = ∫( (rhs_cf*q) )dΩ
    # _liformX((v,q)) = biformX((-sigma_cf,f),(v,q))

    if Dc == 2
      return v -> _liformX(v)
    elseif Dc == 3
      # in 3D, account for the boundary term from IBP
      Γ = BoundaryTriangulation(extrinsic_ambient_model;tags=["bottom_boundary","top_boundary"])
      dΓ = Measure(Γ,degree)
      nΓ = get_normal_vector(Γ)
      boundary((v,q)) = ∫( -f_int*(v⋅nΓ) )dΓ
      return v -> _liformX(v) + boundary(v)
    end
  end


  op = AffineFEOperator(biformX,get_liform(Dc),X,Y)
  A = get_matrix(op)
  b = get_vector(op)
  ns = numerical_setup(symbolic_setup(ls,A),A)
  x = allocate_in_domain(A); fill!(x,0.0)
  solve!(x,ns,b)
  xh = FEFunction(X,x)
  uh,ph = xh

  _e = f - ph
  el2_p = sqrt(sum(∫( _e*_e  )dΩ_error))

  _e = uh - (- sigma_cf ) ### u = -∇p
  el2_u = sqrt(sum(∫( _e⋅_e  )dΩ_error))

 _i_am_main && println("eu = $(el2_u), es = $(el2_p)")

  if return_vtk
    cellfields =  ["u"=> -sigma_cf ,
    "uh"=>uh,
    "eu"=> uh - (-sigma_cf),
    "ph"=>ph, "p"=>f, "e"=>ph-f
                  ]
    writevtk(Ω_ambient,dir*"/extrinsic_ambient_model_nref$(lvl)_p$p_fe",
            cellfields=cellfields,append=false)
  end


  return el2_u, el2_p, false

end


################################################################################
#### Auto convergence test
################################################################################
function main(models::AbstractArray;ps=[2],_i_am_main=true)

  ls = LUSolver()
  dir = @__DIR__
  p_convergence_auto_test(ps,models,hodge_laplacian_scalar,dir,fX,ls;_i_am_main=_i_am_main)
end




end # module
