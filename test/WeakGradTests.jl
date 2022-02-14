function ω(xyz)
  θϕ = xyz2θϕ(xyz)
  θ,ϕ = θϕ
  cos(θ)*cos(ϕ)^2
end

function ωθϕ(θϕ)
  θ,ϕ = θϕ
  cos(θ)*cos(ϕ)^2
end

# spherical surface gradient computed analytically
function gradω(xyz)
  θϕr = xyz2θϕr(xyz)
  θ,ϕ,r = θϕr
  dθ = -sin(θ)*cos(ϕ)
  dϕ = -2*cos(θ)*cos(ϕ)*sin(ϕ)
  spherical_to_cartesian_matrix(θϕr)⋅VectorValue(dθ,dϕ,0)
end

function compute_error_weak_grad(model,order,degree,ls=BackslashSolver())
  # Setup geometry
  Ω=Triangulation(model)
  dΩ=Measure(Ω,degree)
  dω=Measure(Ω,degree,ReferenceDomain())

  # Setup H(div) spaces
  reffe_rt = ReferenceFE(raviart_thomas, Float64, order)
  V = FESpace(model, reffe_rt ; conformity=:HDiv)
  U = TrialFESpace(V)

  # L2 spaces
  reffe_lgn = ReferenceFE(lagrangian, Float64, order)
  S = FESpace(model, reffe_lgn; conformity=:L2)
  R = TrialFESpace(S)

  # Project the analytical field ω onto the L2 space
  a1(u,v) = ∫(v*u)dΩ
  b1(v)   = ∫(v*ω)dΩ
  op      = AffineFEOperator(a1,b1,S,R)
  fels    = LinearFESolver(ls)
  wh      = solve(fels,op)

  # Compute weak grad of wh
  a2(u,v) = ∫(v⋅u)dΩ
  b2(v)   = ∫((-wh)*DIV(v))dω
  op      = AffineFEOperator(a2,b2,U,V)
  fels    = LinearFESolver(ls)
  gradwh  = solve(fels,op)

  e    = gradwh-gradω
  dive = divergence(gradwh)-laplacian_unit_sphere(ωθϕ)∘xyz2θϕ
  sqrt(sum(∫(e⋅e+dive*dive)dΩ))
end
