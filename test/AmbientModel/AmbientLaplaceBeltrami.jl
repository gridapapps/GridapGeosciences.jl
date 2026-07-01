""" Poisson problem using Laplace-Beltrami operator
u + Δᵧ(u) = f
Need to remove the kernal via zeromean FE space
"""

module AmbientLaplaceBeltrami

using Gridap
using Gridap.Helpers
using Gridap.Algebra
using GridapGeosciences
using GridapP4est
using Test


function fX(x)
  x[1]*x[2]*x[3]
end


function laplace_beltrami_solver(
  extrinsic_atlas_model,
  p_fe::Int,dir::String,f::Function,ls=LUSolver(),return_vtk=false;
  _i_am_main=true)


  Dc = num_cell_dims(extrinsic_atlas_model)
  lvl = nref(extrinsic_atlas_model)
  _i_am_main && println("nref = $lvl; p_fe = $p_fe; Dc = $Dc")

  degree = 6*(p_fe+1)
  Ω_ambient = Triangulation(extrinsic_atlas_model)
  dΩ = Measure(Ω_ambient,degree)
  dΩ_error = Measure(Ω_ambient,2*degree)

  V = TestFESpace(Ω_ambient, ReferenceFE(lagrangian,Float64,p_fe); conformity=:H1, constraint=:zeromean)
  U = TrialFESpace(V)

  slap_cf = Δs(f,Ω_ambient)
  rhs_cf = -slap_cf

  @check sum(∫(f)dΩ) < 1e-14 "Function must be zero mean to solve with zeromean FE space!"

  poisson_biform(u,v) =  ∫( gradient(v)⋅gradient(u) )dΩ

  # manufacture rhs functions
  function get_liform(Dc::Int)

    # the manufactured solution
    poisson_liform(v) = ∫(  (rhs_cf*v) )dΩ

    ### This is the other way of mms, where we approximte the LHS operator
    # poisson_liform(v) = ∫( gradient(v)⋅gradient(f) )dΩ

    if Dc == 2
      return v -> poisson_liform(v)
    elseif Dc == 3
      # in 3D, account for the boundary term from IBP
      Γ = BoundaryTriangulation(extrinsic_atlas_model;tags=["bottom_boundary","top_boundary"])
      dΓ = Measure(Γ,degree)
      nΓ = get_normal_vector(Γ)
      f_int = interpolate(f,U)
      boundary(v) = ∫( ( (gradient(f_int) )⋅nΓ)*v )dΓ
      return v -> poisson_liform(v) + boundary(v)
    end
  end

  op = AffineFEOperator(poisson_biform,get_liform(Dc),U,V)

  A = get_matrix(op)
  b = get_vector(op)
  ns = numerical_setup(symbolic_setup(ls,A),A)
  x = allocate_in_domain(A); fill!(x,0.0)
  solve!(x,ns,b)
  uh = FEFunction(U,x)

  _e = f - uh
  e = sqrt(sum(∫( (_e*_e) )dΩ_error))

  if return_vtk
    panel_cfs = [f,uh,f-uh]
    labels = ["u","uh","eu"]
    cellfields = map((x,y) -> x=>y, labels,panel_cfs)
    writevtk(Ω_ambient,dir*"/ambient_model_nref$(lvl)_p$p_fe",
        cellfields=cellfields,append=false)
  end

  return e, false,false
end


################################################################################
#### Auto convergence test
################################################################################
function main(models::AbstractArray;ps=[2],_i_am_main=true)

  ls = LUSolver()
  dir = @__DIR__
  p_convergence_auto_test(ps,models,laplace_beltrami_solver,dir,fX,ls;_i_am_main=_i_am_main)

end





end # module
