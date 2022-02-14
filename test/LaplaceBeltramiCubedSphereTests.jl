# This program solves the Laplace-Beltrami PDE on a unit sphere.
# In strong form, the problem can be stated as:
# find u s.t.
#   -Δ_S(u) = f on \Omega
# Where Δ_S is the spherical Laplacian defined as Δ_S = ∇_S⋅(∇_S(u)),
# with ∇_S() and  ∇_S⋅ being the spherical gradient and spherical
# divergence, resp.

# Given a known analytical solution u(x), the program manufactures f(x)
# by applying -Δ_S to u, and then performs an error convergence analysis.
# For the discretization of the sphere, it uses the so-called cubed sphere
# mesh

function u(x)
  sin(π*x[1])*cos(π*x[2])*exp(x[3])
end
function uθϕ(θϕ)
  u(θϕ2xyz(θϕ))
end
function f(x)
  -(laplacian_unit_sphere(uθϕ)(xyz2θϕ(x)))
end
function Gridap.∇(::typeof(u))
  gradient_unit_sphere(uθϕ)
end

function solve_laplace_beltrami(model,order,degree,ls=BackslashSolver())
    V = FESpace(model,ReferenceFE(lagrangian,Float64,order); conformity=:H1)
    U = TrialFESpace(V)

    Ω = Triangulation(model)
    dΩ = Measure(Ω,degree)

    a(u,v) = ∫(∇(v)⋅∇(u))dΩ
    b(v)   = ∫(v*f)dΩ

    op = AffineFEOperator(a,b,U,V)
    fels = LinearFESolver(ls)
    uh = solve(fels,op)

    e=u-uh
    ∇e = ∇(uh)-∇(u)∘xyz2θϕ

    # H1-norm
    # sqrt(sum(∫( e*e + (∇e)⋅(∇e) )dΩ))
    sqrt(sum(∫( (∇e)⋅(∇e) )dΩ))#,uh
end
