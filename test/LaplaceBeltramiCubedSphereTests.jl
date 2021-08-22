module LaplaceBeltramiCubedSphereTests
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
  using Test
  using Gridap
  import Gridap.Fields: ∇
  using GridapGeosciences
  using Plots

  include("ConvergenceAnalysisTools.jl")

  function u(x)
    sin(π*x[1])*cos(π*x[2])*exp(x[3])
  end
  function uθϕ(θϕ)
    u(θϕ2xyz(θϕ))
  end
  function f(x)
    -(laplacian_unit_sphere(uθϕ)(xyz2θϕ(x)))
  end
  function ∇(::typeof(u))
    gradient_unit_sphere(uθϕ)
  end

  function solve_laplace_beltrami(model,order,degree)
     V = FESpace(model,ReferenceFE(lagrangian,Float64,order); conformity=:H1)
     U = TrialFESpace(V)

     Ω = Triangulation(model)
     dΩ = Measure(Ω,degree)

     a(u,v) = ∫(∇(v)⋅∇(u))dΩ
     b(v)   = ∫(v*f)dΩ

     op = AffineFEOperator(a,b,U,V)
     uh = solve(op)

     e=u-uh
     ∇e = ∇(uh)-∇(u)∘xyz2θϕ

     # H1-norm
     # sqrt(sum(∫( e*e + (∇e)⋅(∇e) )dΩ))
     sqrt(sum(∫( (∇e)⋅(∇e) )dΩ))#,uh
  end

  #model=CubedSphereDiscreteModel(50)
  #e,uh=solve_laplace_beltrami(model,1,8)
  #writevtk(Triangulation(model),"u",nsubcells=4,cellfields=["u"=>u,"uh"=>uh])

  @time ahs1,ak1errors,as1=convergence_study(solve_laplace_beltrami,generate_n_values(2),1,2)
  @test as1 ≈ 1.0040693202861342

  @time ahs2,ak2errors,as2=convergence_study(solve_laplace_beltrami,generate_n_values(2),2,4)
  @test as2 ≈ 2.045298259079679


  #  plot([ahs1,ahs2],[ak1errors,ak2errors],
  #  xaxis=:log, yaxis=:log,
  #  label=["k=1 analytical map" "k=2 analytical map"],
  #  shape=:auto,
  #  xlabel="h",ylabel="error H1 semi-norm", legend=:bottomright)

end
