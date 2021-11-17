module CubedSphereDiscreteModelsTests
 using GridapGeosciences
 using Gridap

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
 function convergence_discrete_cubed_sphere_surface(hs,model_args_series,degree)
   approx_surfaces=Float64[]
   rel_errors=Float64[]
   exact_surface=4π
   current=1
   for args in model_args_series
      model=CubedSphereDiscreteModel(args...)
      approx_surface=compute_discrete_cubed_sphere_surface(model,degree)
      append!(approx_surfaces,approx_surface)
      append!(rel_errors,abs(approx_surface-exact_surface)/exact_surface)
      current=current*10
   end
   return approx_surfaces,rel_errors,slope(hs,rel_errors)
 end

 function compute_discrete_cubed_sphere_surface(model,degree)
   Ω = Triangulation(model)
   dΩ = Measure(Ω,degree)
   f(x)=1
   sum(∫(f)dΩ)
 end

end
