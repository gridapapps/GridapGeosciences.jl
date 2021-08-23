module WeakGradTests
   using Gridap
   using GridapGeosciences
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
   function dω(xyz)
    θϕr = xyz2θϕr(xyz)
    θ,ϕ,r = θϕr
    dθ = -sin(θ)*cos(ϕ)
    dϕ = -2*cos(θ)*cos(ϕ)*sin(ϕ)
    spherical_to_cartesian_matrix(θϕr)⋅VectorValue(dθ,dϕ,0)
  end

   function ∇(::typeof(ωθϕ))
    #gradient_unit_sphere(ωθϕ)
    dω
   end



   function compute_error_weak_grad(model,order,degree)
     # Setup geometry
     Ω=Triangulation(model)
     dΩ=Measure(Ω,degree)
     dω=Measure(Ω,degree,ReferenceDomain())
     n=get_normal_vector(model)

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

     e = wh-∇(ωθϕ)
     sqrt(sum(∫(e⋅e)dΩ)),wh
   end
   model=CubedSphereDiscreteModel(50)
   e,wh=compute_error_weak_grad(model,1,12)
   writevtk(Triangulation(model),"XXX",cellfields=["exact"=>∇(ωθϕ),"approx"=>wh, "error"=>wh-∇(ωθϕ)])


   @time ahs0,ak0errors,as0=convergence_study(compute_error_weak_grad,generate_n_values(2),1,4)
   @test as0 ≈ 2.0842745262542386

   @time bihs0,bik0errors,bis0=convergence_study(compute_error_weak_grad,generate_n_values(2),1,0,4)
   @test bis0 ≈ 0.9474505144846311

   @time biqhs0,biqk0errors,biqs0=convergence_study(compute_error_weak_grad,generate_n_values(2),2,0,4)
   @test biqs0 ≈ 2.08294675561341


  #  plotd=plot([ahs0,bihs0,biqhs0],[ak0errors,bik0errors,biqk0errors],
  #  xaxis=:log, yaxis=:log,
  #  label=["k=0 analytical map" "k=0 bilinear map" "k=0 biquadratic map"],
  #  shape=:auto,
  #  xlabel="h",ylabel="L2 error norm", legend=:bottomright)

  #  savefig(plotd,"L2_error_weak_div_perp.png")

end
