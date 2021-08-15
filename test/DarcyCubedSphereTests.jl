module DarcyCubedSphereTests

# This program solves Problem in Section 5.1.1 in the following paper
# Rognes, M. E., Ham, D. A., Cotter, C. J., and McRae, A. T. T.
# Automating the solution of PDEs on the sphere and other manifolds in FEniCS 1.2,
# Geosci. Model Dev., 6, 2099–2119, https://doi.org/10.5194/gmd-6-2099-2013, 2013.

using Test
using Gridap
using GridapGeosciences
using Plots

include("ConvergenceAnalysisTools.jl")

function solve_darcy(n,order,degree)
  g(x) = sin(0.5*π*x[2]) # x[1]*x[2]*x[3]
  model = CubedSphereDiscreteModel(n,2)
  RT=ReferenceFE(raviart_thomas,Float64,order)
  DG=ReferenceFE(lagrangian,Float64,order)
  V = FESpace(model,RT; conformity=:Hdiv)
  Q = FESpace(model,DG; conformity=:L2)
  T = FESpace(model,DG; conformity=:L2)

  U = TrialFESpace(V)
  P = TrialFESpace(Q)
  R = TrialFESpace(T)

  Y = MultiFieldFESpace([V, Q, T])
  X = MultiFieldFESpace([U, P, R])

  trian = Triangulation(model)
  degree = 2
  dΩ = Measure(trian,degree)
  dω = Measure(trian,degree,ReferenceDomain())

  a((u,p,r),(v,q,t)) = ∫( v⋅u + q*r + t*p )*dΩ +  ∫(q*DIV(u)+DIV(v)*p)*dω
  awrong((u,p,r),(v,q,t)) = ∫( v⋅u + q*r + t*p + q*(∇⋅u) + (∇⋅v)*p)*dΩ
  b((v,q,t)) = ∫(q*g)*dΩ

  op = AffineFEOperator(a,b,X,Y)
  xh = solve(op)
  uh, ph, rh = xh
  # writevtk(trian, "u", cellfields=["uh"=>uh])
  e = g-rh
  sqrt(sum(∫(e*e)dΩ))
end

"""
Performs a convergence study.

Returns a tuple with three entries.
 [1] Values of h.
 [2] Relative errors.
 [3] Slope of the log(h)-log(err) relative error convergence curve.
"""
function convergence_darcy_cubed_sphere(n_values,order,degree)
  hs=Float64[]
  errors=Float64[]
  for n in n_values
     err=solve_darcy(n,order,degree)
     append!(errors,err)
  end
  println(n_values)
  hs=[2.0/n for n in n_values]
  return hs,errors,slope(hs,errors)
end


hs,k0errors,s=convergence_darcy_cubed_sphere(generate_n_values(2),0,2)
@test s ≈ 1.0062026869499259
err=solve_darcy(10,1,4)
@test err < 1.e-14

plot(hs,[k0errors],
    xaxis=:log, yaxis=:log,
    label=["L2 k=0"],
    shape=:auto,
    xlabel="h",ylabel="L2 error norm")

end # module
