  using Gridap
  using GridapGeosciences

  p_ex(x) = -x[1]x[2]x[3]
  u_ex(x) = VectorValue(
    x[2]x[3] - 3x[1]x[1]x[2]x[3] / (x[1]x[1] + x[2]x[2] + x[3]x[3]),
    x[1]x[3] - 3x[1]x[2]x[2]x[3] / (x[1]x[1] + x[2]x[2] + x[3]x[3]),
    x[1]x[2] - 3x[1]x[2]x[3]x[3] / (x[1]x[1] + x[2]x[2] + x[3]x[3])
    )
#   function uθϕ_ex(θϕ)
#     u_ex(θϕ2xyz(θϕ))
#   end
#   function pθϕ_ex(θϕ)
#     p_ex(θϕ2xyz(θϕ))
#   end
#   function Gridap.Fields.gradient(::typeof(p_ex))
#      gradient_unit_sphere(pθϕ_ex)∘xyz2θϕ
#   end
#   function Gridap.Fields.divergence(::typeof(u_ex))
#     divergence_unit_sphere(uθϕ_ex)∘xyz2θϕ
#   end
  f_ex(x) = -12x[1]x[2]x[3] # divergence(u_ex)(x)
  g_ex(x) = u_ex(x)+gradient(p_ex)(x)

  # using Distributions
  # θϕ=Point(rand(Uniform(0,2*pi)),rand(Uniform(-pi/2,pi/2)))
  # θϕ2xyz(θϕ)

  function assemble_darcy_problem(model, order)
    rt_reffe = ReferenceFE(raviart_thomas, Float64, order)
    lg_reffe = ReferenceFE(lagrangian, Float64, order)
    V = FESpace(model, rt_reffe, conformity=:Hdiv)
    U = TrialFESpace(V)

    Q = FESpace(model, lg_reffe; conformity=:L2)
    P = TrialFESpace(Q)

    # X = MultiFieldFESpace([U, P])
    # Y = MultiFieldFESpace([V, Q])

    Ω = Triangulation(model)
    degree = 10 # 2*order
    dΩ = Measure(Ω, degree)
    dω = Measure(Ω,degree,ReferenceDomain())

    a11(u,v)=∫(v⋅u)dΩ
    B11=assemble_matrix(a11,U,V)

    a12(p,v)=∫(∇⋅v*p)dΩ
    B12=assemble_matrix(a12,P,V)

    a21(u,q)=∫(∇⋅u*q)dΩ
    B21=assemble_matrix(a12,P,V)
 
    a22(p,q)=∫(p*q)dΩ
    B22=assemble_matrix(a22,P,Q)    

    # a((u, p), (v, q)) = ∫(v ⋅ u + p*q)dΩ + ∫(∇⋅(u)*q -∇⋅(v)*p)dΩ
    # l((v, q)) = ∫(q*f_ex + q*p_ex)dΩ
    # AffineFEOperator(a, l, X, Y)
    B11, B12, B21, B22
  end

  function solve_darcy_problem(op)
    solve(op)
  end 

  function compute_darcy_errors(model, order, xh)
    uh,ph=xh 
    eph = ph-p_ex
    euh = uh-u_ex
    degree=2*order
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    err_p = sum(∫(eph*eph)dΩ)
    err_u_l2  = sum(∫(euh⋅euh)dΩ)
    err_u_div = sum(∫(euh⋅euh + (∇⋅(euh))*(∇⋅(euh)))dΩ)
    err_p, err_u_l2, err_u_div
    surface=sum(∫(1)dΩ)
    surface, surface, surface 
  end 

order=0
n=1
model = CubedSphereDiscreteModel(n ; radius=1.0)
B211,B212,B221,B222=assemble_darcy_problem(model, order);

for geom_order in 1:8
  order=0
  model = CubedSphereDiscreteModel(n, geom_order; radius=1.0)
  B111,B112,B121,B122=assemble_darcy_problem(model, order)
  println(norm(B111-B211)/norm(B211), " ", 
          norm(B112-B212)/norm(B212), " ", 
          norm(B121-B221)/norm(B221), " ", 
          norm(B122-B222)/norm(B222))
  # println(B1)
end 

#for n in [4,8,16,25,40]
for n in [1,]
  model = CubedSphereDiscreteModel(n; radius=1.0)
  op=assemble_darcy_problem(model, order)
  println(Array(op.op.matrix))
  # xh=solve_darcy_problem(op)
  # err_p, err_u_l2, err_u_div = compute_darcy_errors(model, order, xh)
  # println(err_p, " ", err_u_l2, " ", err_u_div)
end 



