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
 @test round(s,digits=3) ≈ 1.986
 _,_,s=convergence_discrete_cubed_sphere_surface(generate_n_values(2),2,4)
 @test round(s,digits=3) ≈ 4.023

 model=CubedSphereDiscreteModel(10,1;radius=rₑ)
 ψₖ = get_cell_map(model)
 @test norm(evaluate(ψₖ[rand(1:num_cells(model))],Point(1,1))) ≈ rₑ

 model=CubedSphereDiscreteModel(10;radius=rₑ)
 ψₖ = get_cell_map(model)
 @test norm(evaluate(ψₖ[rand(1:num_cells(model))],Point(rand(),rand()))) ≈ rₑ

 function detJtJ(Jt::Gridap.Fields.MultiValue{Tuple{D1,D2}}) where {D1,D2}
  J = transpose(Jt)
  sqrt(det(Jt⋅J))
 end

 function JtJ(Jt::Gridap.Fields.MultiValue{Tuple{D1,D2}}) where {D1,D2}
  J = transpose(Jt)
  Jt⋅J
 end

 function detJJt(Jt::Gridap.Fields.MultiValue{Tuple{D1,D2}}) where {D1,D2}
  J = transpose(Jt)
  sqrt(abs(det(J⋅Jt)))
 end

 function JJt(Jt::Gridap.Fields.MultiValue{Tuple{D1,D2}}) where {D1,D2}
  J = transpose(Jt)
  J⋅Jt
 end

 Ω = Triangulation(model)
 dΩ = Measure(Ω,2)

 vdetJtJ   = lazy_map(Operation(detJtJ),lazy_map(Broadcasting(∇),get_cell_map(model)))
 cfdetJtJ  = Gridap.CellData.GenericCellField(vdetJtJ,Ω,ReferenceDomain())
 p=get_cell_points(dΩ.quad)
 detJtJp=cfdetJtJ(p)

 println(detJtJp[1][1])
 println(detJtJp[1][2])
 println(detJtJp[1][3])
 println(detJtJp[1][4])

 vJtJ   = lazy_map(Operation(JtJ),lazy_map(Broadcasting(∇),get_cell_map(model)))
 cfJtJ  = Gridap.CellData.GenericCellField(vJtJ,Ω,ReferenceDomain())
 p=get_cell_points(dΩ.quad)
 JtJp=cfJtJ(p)

 println(JtJp[1][1])
 println(JtJp[1][2])
 println(JtJp[1][3])
 println(JtJp[1][4])

 vdetJJt   = lazy_map(Operation(detJJt),lazy_map(Broadcasting(∇),get_cell_map(model)))
 cfdetJJt  = Gridap.CellData.GenericCellField(vdetJJt,Ω,ReferenceDomain())
 p=get_cell_points(dΩ.quad)
 detJJtp=cfdetJJt(p)

 println(detJJtp[1][1])
 println(detJJtp[1][2])
 println(detJJtp[1][3])
 println(detJJtp[1][4])


 vJJt   = lazy_map(Operation(JJt),lazy_map(Broadcasting(∇),get_cell_map(model)))
 cfJJt  = Gridap.CellData.GenericCellField(vJJt,Ω,ReferenceDomain())
 p=get_cell_points(dΩ.quad)
 JJtp=cfJJt(p)

 println(JJtp[1][1])
 println(JJtp[1][2])
 println(JJtp[1][3])
 println(JJtp[1][4])

 vJt = lazy_map(Broadcasting(∇),get_cell_map(model))
 cfJt  = Gridap.CellData.GenericCellField(vJt,Ω,ReferenceDomain())
 p=get_cell_points(dΩ.quad)
 Jtp=cfJt(p)

 println(Jtp[1][1])
 println(Jtp[1][2])
 println(Jtp[1][3])
 println(Jtp[1][4])



end
