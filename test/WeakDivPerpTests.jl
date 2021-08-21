module WeakDivPerpTests
   using Gridap
   using GridapGeosciences
   using Plots
   using Test

   include("ConvergenceAnalysisTools.jl")


   # Nondivergent velocity field tangent to the sphere
   # Reference:
   #     Lauritzen et. al. GMD 2012
   #     Eq. (18) and (19)
   function W(θϕ)
     θ,ϕ = θϕ
     u = 2*sin(θ)^2*sin(2*ϕ) + 2π/5*cos(ϕ)
     v = 2*sin(2*θ)*cos(ϕ)
     spherical_to_cartesian_matrix(VectorValue(θ,ϕ,1.0))⋅VectorValue(u,v,0)
   end

   """
   Spherical divergence on unit sphere applied to perp(W)
   """
   function divM_perpW(θϕ)
     n_times_W=(θϕ)->(normal_unit_sphere(θϕ)×W(θϕ))
     f=divergence_unit_sphere(n_times_W)
     f(θϕ)
   end

   function compute_error_weak_div_perp(model,order,degree)
     # Setup geometry
     Ω=Triangulation(model)
     dΩ=Measure(Ω,degree)
     n=get_normal_vector(model)

     # Setup H(div) spaces
     reffe_rt = ReferenceFE(raviart_thomas, Float64, order)
     V = FESpace(model, reffe_rt ; conformity=:HDiv)
     U = TrialFESpace(V)

     # H1 spaces
     reffe_lgn = ReferenceFE(lagrangian, Float64, order+1)
     S = FESpace(model, reffe_lgn; conformity=:H1)
     R = TrialFESpace(S)

     # Project the analytical velocity field W onto the H(div) space
     a1(u,v) = ∫(v⋅u)dΩ
     b1(v)   = ∫(v⋅(W∘xyz2θϕ))dΩ
     op      = AffineFEOperator(a1,b1,U,V)
     wh      = solve(op)

     e = (W∘xyz2θϕ)-wh
     println(sqrt(sum(∫(e⋅e)dΩ)))

     # Compute weak div_perp operator
     a2(u,v) = ∫(v*u)dΩ
     b2(v)   = ∫(⟂(∇(v),n)⋅(wh) )dΩ
     op      = AffineFEOperator(a2,b2,R,S)
     divwh   = solve(op)
     e = divwh-divM_perpW∘xyz2θϕ

     sqrt(sum(∫(e*e)dΩ))
   end
   @time ahs0,ak0errors,as0=convergence_study(compute_error_weak_div_perp,generate_n_values(2),0,4)
   @test as0 ≈ 2.0842745262542386

   @time bihs0,bik0errors,bis0=convergence_study(compute_error_weak_div_perp,generate_n_values(2),1,0,4)
   @test bis0 ≈ 0.9474505144846311

   @time biqhs0,biqk0errors,biqs0=convergence_study(compute_error_weak_div_perp,generate_n_values(2),2,0,4)
   @test biqs0 ≈ 2.08294675561341


  #  plotd=plot([ahs0,bihs0,biqhs0],[ak0errors,bik0errors,biqk0errors],
  #  xaxis=:log, yaxis=:log,
  #  label=["k=0 analytical map" "k=0 bilinear map" "k=0 biquadratic map"],
  #  shape=:auto,
  #  xlabel="h",ylabel="L2 error norm", legend=:bottomright)

  #  savefig(plotd,"L2_error_weak_div_perp.png")

end
