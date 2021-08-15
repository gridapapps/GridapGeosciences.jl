module CubedSphereDiscreteModelsTests
 using GridapGeosciences
 using Gridap
 using Test

 include("ConvergenceAnalysisTools.jl")

 """
 Performs a convergence study of the discrete cubed sphere surface
 using a Gauss quadrature of degree degree for numerical integration.
 order is the order of the polynomial space used to approximate the
 surface of the cubed sphere.

 Returns a tuple with three entries.
  [1] Discrete cubed sphere surfaces for the associated values of n.
  [2] Relative errors among the approximate surface and the exact one.
  [3] Slope of the log(h)-log(err) relative error convergence curve.
 """
 function convergence_discrete_cubed_sphere_surface(n_values,order,degree)
   approx_surfaces=Float64[]
   rel_errors=Float64[]
   exact_surface=4π
   current=1
   for n in n_values
      approx_surface=compute_discrete_cubed_sphere_surface(n,order,degree)
      append!(approx_surfaces,approx_surface)
      append!(rel_errors,abs(approx_surface-exact_surface)/exact_surface)
      current=current*10
   end
   println(n_values)
   hs=[2.0/n for n in n_values]
   return approx_surfaces,rel_errors,slope(hs,rel_errors)
 end

 function compute_discrete_cubed_sphere_surface(n,order,degree)
   model = CubedSphereDiscreteModel(n,order)
   Ω = Triangulation(model)
   dΩ = Measure(Ω,degree)
   f(x)=1
   sum(∫(f)dΩ)
 end

 _,_,s=convergence_discrete_cubed_sphere_surface(generate_n_values(2),1,0)
 @test s ≈ 1.960928154802903
 _,_,s=convergence_discrete_cubed_sphere_surface(generate_n_values(2),2,4)
 @test s ≈ 4.0372255440047615
end
