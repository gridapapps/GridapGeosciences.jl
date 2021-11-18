# This program solves Problem in Section 5.1.1 in the following paper
# Rognes, M. E., Ham, D. A., Cotter, C. J., and McRae, A. T. T.
# Automating the solution of PDEs on the sphere and other manifolds in FEniCS 1.2,
# Geosci. Model Dev., 6, 2099–2119, https://doi.org/10.5194/gmd-6-2099-2013, 2013.

function solve_darcy(model,order,degree,ls=BackslashSolver())
  g(x) = sin(0.5*π*x[2]) # x[1]*x[2]*x[3]
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
  dΩ = Measure(trian,degree)
  dω = Measure(trian,degree,ReferenceDomain())

  aref((u,p,r),(v,q,t)) = ∫( v⋅u + q*r + t*p )*dΩ +  ∫(q*DIV(u)+DIV(v)*p)*dω
  aphy((u,p,r),(v,q,t)) = ∫( v⋅u + q*r + t*p + q*(∇⋅u) + (∇⋅v)*p)*dΩ
  b((v,q,t)) = ∫(q*g)*dΩ

  op = AffineFEOperator(aref,b,X,Y)
  fels = LinearFESolver(ls)
  xh = solve(fels,op)
  uh, ph, rh = xh
  e = g-rh
  sqrt(sum(∫(e*e)dΩ))
end
