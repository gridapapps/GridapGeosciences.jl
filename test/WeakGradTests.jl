module WeakGradTests
   using Gridap
   using GridapGeosciences
   import Gridap.Fields: ∇, divergence
   using Plots
   using Test

   include("ConvergenceAnalysisTools.jl")

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

  #  function divergence(::typeof(ωθϕ))
  #   divergence_unit_sphere(ωθϕ)
  #  end

   function compute_error_weak_grad(model,order,degree)
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
     wh      = solve(op)

     # Compute weak grad of wh
     a2(u,v) = ∫(v⋅u)dΩ
     b2(v)   = ∫((-wh)*DIV(v))dω
     op      = AffineFEOperator(a2,b2,U,V)
     gradwh  = solve(op)

     e    = gradwh-gradω
     dive = divergence(gradwh)-laplacian_unit_sphere(ωθϕ)∘xyz2θϕ
     sqrt(sum(∫(e⋅e+dive*dive)dΩ))
   end
   #model=CubedSphereDiscreteModel(24)
   #e,gradwh=compute_error_weak_grad(model,1,4)
   #writevtk(Triangulation(model),"RT124x24analyticalmap",cellfields=["gradwh"=>gradwh])

   @time ahs0,ak0errors,as0=convergence_study(compute_error_weak_grad,generate_n_values(2),0,4)
   @test as0 ≈ 0.8346320885900106

   @time ahs1,ak1errors,as1=convergence_study(compute_error_weak_grad,generate_n_values(2,n_max=50),1,8)
   @test as1 ≈ 1.1034326200306834

  #  plotd=plot([ahs0,ahs1],[ak0errors,ak1errors],
  #  xaxis=:log, yaxis=:log,
  #  label=["k=0 analytical map" "k=1 analytical map"],
  #  shape=:auto,
  #  xlabel="h",ylabel="H(div) error norm", legend=:bottomright)

  #  savefig(plotd,"L2_error_weak_div_perp.png")

end
