# # Shallow water on the cubed sphere manifold (intrinsic formulation)
#
# This example solves the shallow water equations.
# The evolution of the prognostic variables, $\widetilde{\boldsymbol{u}}$ and $\widetilde{\varphi}$ is:
#
# ```math
# \begin{align*}
# \partial_t \widetilde{\boldsymbol{u}}
# + \widetilde{q} \widetilde{\boldsymbol{F}}^\dagger
# + \nabla_{\gamma} \widetilde{\Phi}  &=  0  \quad \text{in} \quad \gamma\times[0,T],
# \\
# \partial_t \widetilde{\varphi}
# + \nabla_{\gamma}\cdot \widetilde{\boldsymbol{F}} &= 0
# \quad  \quad \text{in} \quad \gamma\times[0,T],
# \end{align*}
# ```
#
# The diagnostic variables solve algebraic constraints:
# ```math
# \begin{align*}
# \widetilde{\boldsymbol{F}} &= \widetilde{\varphi}\widetilde{\boldsymbol{u}} , \\
# \widetilde{\Phi} &= \frac{1}{2} \widetilde{\boldsymbol{u}}\cdot\widetilde{\boldsymbol{u}} + g_r \widetilde{\varphi}  ,\\
# \widetilde{q} &= \frac{1 }{\widetilde{\varphi}}(\nabla_{\gamma}^\dagger\cdot \widetilde{\boldsymbol{u}}  + f) ,
# \end{align*}
# ```
#
# where $\gamma$ is the cubed sphere manifold,  $g_r$ is the acceleration due to gravitiy, and $f$ is the Coriolis force.
# We use compatible finite element pairings to solve in the parametric space of the cubed sphere.
# This problem is a differential algebraic equation (DAE), which is solved using a bespoke
# ``DAEFEOperator`` that builds on Gridap's ODE API.

# ## Set up
# First load all required packages. In this example, we will use a distributed model
# and iterative solvers. So we initialise MPI.
using GridapGeosciences
using Gridap
using Gridap.Algebra
using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapDistributed
using PartitionedArrays
using MPI

MPI.Init()
ranks = distribute_with_mpi(LinearIndices((prod(MPI.Comm_size(MPI.COMM_WORLD)),)))

# ## Discrete model
# To obtain a refined 2D parametric model, we pass $\ell$ levels of refinement:
ℓ = 2
radius = 6371220.0
coarse_mesh = CubedSphereMesh(radius)
omodel = AtlasOctreeDistributedDiscreteModel(ranks, coarse_mesh, ℓ; manifold_style=IntrinsicManifold())

# ## Triangulation
# Now we extract the triangulated and the panel ids associated to each cell:
Ω = Triangulation(omodel)

# ## FE Spaces
# Trial and test spaces relevant to compatible finite element pairings are defined
# using Gridap's high level API (refer to [Cotter et al. 2012](https://doi.org/10.1016/j.jcp.2012.05.020)).
order = 1
R = TestFESpace(Ω, ReferenceFE(lagrangian,Float64,order+1); conformity=:H1)
#H = TransientTrialFESpace(R)
H = TrialFESpace(R)

V = TestFESpace(Ω, ReferenceFE(raviart_thomas,Float64,order); conformity=:HDiv)
#U = TransientTrialFESpace(V)
U = TrialFESpace(V)

Q = TestFESpace(Ω, ReferenceFE(lagrangian,Float64,order); conformity=:L2)
#P = TransientTrialFESpace(Q)
P = TrialFESpace(Q)

# The multifield FE spaces for the prognostic and diagnostic variables are:
X_prog = MultiFieldFESpace([U,P]) # u, p
Y_prog = MultiFieldFESpace([V,Q]) # u, p

X_diag = MultiFieldFESpace([H,U,P]) # q, F, Φ
#Y_diag = MultiFieldFESpace([R,V,Q]) # q, F, Φ

# ## Initial conditions
# The initial condition for velocity, fluid depth and Coriolis are defined
# as per the Williamson 2 test case, in [Williamson et al. 1992](https://doi.org/10.1016/S0021-9991(05)80016-6).
#
# The dimensional parameters are:
aₑ = radius # m
gₑ = 9.80616 # m/s^{-2}
ωₑ = 7.29e-5 # s^{-1}
Hₑ = 2.94e4/gₑ # m
uₑ = 2*π*aₑ/(12*24*3600) # m s^{-1}

# Consider a length scale of $a_{e}$ and time scale of $1/\omega$ yields the non-dimensional parameters:
gravity = gₑ#*(1/ωₑ)^2/aₑ
ω = ωₑ#*(1/ωₑ)
H₀ = Hₑ#/aₑ
u₀ = uₑ#/aₑ*(1/ωₑ)

# The initial conditions are defined as a function of the forward map as follows:
function u0(xyz)
  ζ = 0.0
  θϕr   = xyz2θϕr(xyz)
  θ,ϕ,r = θϕr
  u     = u₀*(cos(ϕ)*cos(ζ) + cos(θ)*sin(ϕ)*sin(ζ))
  v     = - u₀*sin(θ)*sin(ζ)
  function _spherical_to_cartesian_matrix(θϕr)
    θ,ϕ,r = θϕr
    TensorValue(-sin(θ)       , cos(θ)       ,      0,
                -sin(ϕ)*cos(θ),-sin(ϕ)*sin(θ), cos(ϕ),
                 cos(ϕ)*cos(θ), cos(ϕ)*sin(θ), sin(ϕ))
  end
  _spherical_to_cartesian_matrix(θϕr)⋅VectorValue(u,v,0)
end

function h0(xyz)
  ζ = 0.0
  θϕr = xyz2θϕr(xyz)
  θ,ϕ,r = θϕr
  h  = -cos(θ)*cos(ϕ)*sin(ζ) + sin(ϕ)*cos(ζ)
  H₀- (ω*u₀  + 0.5*u₀*u₀)*h*h/gravity
end

function f0(xyz)
  ζ = 0.0
  θϕr = xyz2θϕr(xyz)
  θ,ϕ,r = θϕr
  2.0*ω*( -cos(θ)*cos(ϕ)*sin(ζ) + sin(ϕ)*cos(ζ) )
end

# ## Weak form
# To define the weak form, we require the metric and measure, as well as the
# the matrix that represents the perp operator.
# We use an increased degree of quadrature to exactly approximate the geometrical map included in the weak form.
ambient_map_cf = AmbientMapCellField(Ω)
g = MetricCellField(Ω)
ginv = InvMetricCellField(Ω)
meas = MeasureCellField(Ω)
covariant_basis_cf = transpose∘∇(ambient_map_cf)
Aperp = [0.0 -1.0
        1.0 0.0]
Rperp = TensorValue(Aperp)
Rperp_cf = CellField(Rperp,Ω)
dΩ = Measure(Ω,3*(order+1))

# Then converted into a cellfield, where we extract the contravariant components
# for the velocity:
u_cf = meas*((pinvJ∘transpose∘∇(ambient_map_cf))⋅(u0∘ambient_map_cf))
h_cf = h0∘ambient_map_cf
f_cf = f0∘ambient_map_cf

# ### Diagnostic variables
# The weak forms for the diagnostic variables are:
#resq(((u,p),(q,F,Φ)),(w,v,ψ)) = ∫( q*p*w*meas )dΩ - ∫( f_cf*w*meas )dΩ - ∫( (((Rperp_cf⋅u)⋅ginv)⋅∇(w))*meas)dΩ
#resF(((u,p),(q,F,Φ)),(w,v,ψ)) = ∫( (F⋅(g⋅v))*(1/meas) )dΩ - ∫( p*(u⋅(g⋅v))*(1/meas))dΩ
#resΦ(((u,p),(q,F,Φ)),(w,v,ψ)) = ∫( Φ*ψ*meas )dΩ - ∫( gravity*p*ψ*meas  )dΩ - ∫( 0.5*(u⋅(g⋅u))ψ*(1/meas))dΩ

#res_y(t,((u,p),(q,F,Φ)),(w,v,ψ)) = resq(((u,p),(q,F,Φ)),(w,v,ψ)) + resF(((u,p),(q,F,Φ)),(w,v,ψ)) + resΦ(((u,p),(q,F,Φ)),(w,v,ψ))
#jac_y(t,((u,p),(q,F,Φ)),(dq,dF,dΦ),(w,v,ψ)) = ∫( dq*p*w*meas )dΩ + ∫( (dF⋅(g⋅v))*(1/meas) )dΩ + ∫( dΦ*ψ*meas  )dΩ

# ### Prognostic variables
# The weak forms for the prognostic variables are as follows, where we use
# SUPG stabilisation in the velocity equation
#res_p(((u,p),(q,F,Φ)),(v,r),(q0,F0,Φ0)) = ∫( r*(∇⋅F) )dΩ
#res_u(((u,p),(q,F,Φ)),(v,r),(q0,F0,Φ0)) = ( ∫( ( q*( (Rperp_cf⋅F)⋅v)))dΩ
#                              + ∫( -τ*( (q-q0)/dt)*( (Rperp_cf⋅F)⋅v))dΩ
#                              + ∫( -τ*(u⋅∇(q))*( (Rperp_cf⋅ F)⋅v)*(1/meas))dΩ
#                              - ∫( Φ*(∇⋅v) )dΩ )

#mass(t,(dut,dpt),(v,r)) = ∫( (dut⋅(g⋅v))*(1/meas) )dΩ + ∫( (dpt*r)*meas )dΩ
#res_x(t,((u,p),(q,F,Φ)),(v,r),(q0,F0,Φ0)) = res_u(((u,p),(q,F,Φ)),(v,r),(q0,F0,Φ0)) + res_p(((u,p),(q,F,Φ)),(v,r),(q0,F0,Φ0))
#jac_x(t,((u,p),(q,F,Φ)),(du,dp),(v,r),(q0,F0,Φ0)) = ∫( -τ*(du⋅∇(q))*( (Rperp_cf⋅F)⋅v)*(1/meas))dΩ
#jac_xt(t,((u,p),(q,F,Φ)),(dut,dpt),(v,r),(q0,F0,Φ0)) =  ∫( (dut⋅(g⋅v))*(1/meas) )dΩ + ∫((dpt*r)*meas )dΩ

# ## Transient problem
# The transient problem requires a linear solver for both the prognostic and diagnostic problem.
# In this example, we using a Conjugate Gradient method for both.
#ls_diag = CGSolver(JacobiLinearSolver();rtol=1e-12,verbose=i_am_main(ranks),name="diagnostic_solver")
#ls_ode = CGSolver(JacobiLinearSolver();rtol=1e-12,verbose=i_am_main(ranks),name="ode_solver")

# The transient parameters are:
t0 = 0.0
tF = 12.0*24.0*3600.0
nsteps = 100
dt = tF/nsteps
τ = dt/2

# The transient operator, DAE operator and transient solution, integrated using SSPRK3, is defined as follows:
#opT = TransientSemilinearFEOperator(mass,res_x,(jac_x,jac_xt),X_prog,Y_prog)
#opFE = FEOperator(res_y,jac_y,X_diag,Y_diag)
#opDAE = DAEFEOperator(opT,opFE,ls_diag)
#ode_solver = RungeKutta(ls_ode,ls_ode,dt,:EXRK_SSP_3_3)
#xh0 = interpolate_everywhere([u_cf,h_cf],X_prog)
#solT  = solve(ode_solver,opDAE,t0,tF,xh0)

x = interpolate_everywhere([u_cf,h_cf],X_prog)
xd = interpolate_everywhere([f_cf,u_cf,h_cf],X_diag)
# ## Post processing
# The diagnostic solution can be extracted from the cache of the ode solution.
# To do so, we utilise the iterable functionality of ``solT`` as follows:

i_am_main(ranks) && mkpath("shallow_water_sol")
PartitionedArrays.barrier(ranks)

# Assemble the operators
λ = 0.5
Jmat((u,p),(v,q)) = ∫( (u⋅(g⋅v))*(1/meas) )dΩ + ∫(p*q*meas)dΩ + ∫(dt*λ*f_cf*((Rperp_cf⋅u)⋅v))dΩ - ∫(dt*λ*gravity*(p*(∇⋅v)))dΩ + ∫(dt*λ*H₀*(q*(∇⋅u)))dΩ
J = assemble_matrix(Jmat, X_prog, Y_prog)

Mumat(u,v) = ∫( (u⋅(g⋅v))*(1/meas) )dΩ
Mpmat(p,q) = ∫( (p*q)*meas )dΩ
Mu = assemble_matrix(Mumat, U, V)
Mp = assemble_matrix(Mpmat, P, Q)

cgu = CGSolver(;rtol=1.0e-12)
cgu_ns = numerical_setup(symbolic_setup(cgu,Mu),Mu)
cgp = CGSolver(;rtol=1.0e-12)
cgp_ns = numerical_setup(symbolic_setup(cgp,Mp),Mp)
gmres = GMRESSolver(100;restart=true,rtol=1.0e-12)
gmres_ns = numerical_setup(symbolic_setup(gmres,J),J)

function diagnostics!(q,F,Φ,u,h,dΩ,meas,g,ginv,U,P,H,R,cgu_ns,cgh_ns,f_cf,Rperp_cf,Mu,Mh)
  buvec(v) = ∫( h*(u⋅(g⋅v))*(1/meas) )dΩ
  bu = assemble_vector(buvec, U)
  bus = allocate_in_domain(Mu)
  copy!(bus, bu)
  Fx = allocate_in_domain(Mu)
  fill!(Fx, 0.0)
  solve!(Fx, cgu_ns, bus)
  consistent!(Fx) |> wait
  get_free_dof_values(F) .= Fx
  consistent!(get_free_dof_values(F)) |> wait

  bhvec(p) = ∫( gravity*h*p*meas )dΩ + ∫( 0.5*(u⋅(g⋅u))*p*(1/meas))dΩ
  bh = assemble_vector(bhvec, P)
  bhs = allocate_in_domain(Mh)
  copy!(bhs, bh)
  Φx = allocate_in_domain(Mh)
  fill!(Φx, 0.0)
  solve!(Φx, cgh_ns, bhs)
  consistent!(Φx) |> wait
  get_free_dof_values(Φ) .= Φx
  consistent!(get_free_dof_values(Φ)) |> wait

  Mwmat(r,s) = ∫( r*s*h*meas )dΩ
  Mw = assemble_matrix(Mwmat, H, R)
  cgw = LinearSolvers.CGSolver(;rtol=1.0e-8)
  cgw_ns = numerical_setup(symbolic_setup(cgu,Mw),Mw)
  bwvec(r) = ∫( f_cf*r*meas )dΩ + ∫( (((Rperp_cf⋅u)⋅ginv)⋅∇(r))*meas)dΩ
  bw = assemble_vector(bwvec, H)
  bws = allocate_in_domain(Mw)
  copy!(bws, bw)
  qx = allocate_in_domain(Mw)
  fill!(qx, 0.0)
  solve!(qx, cgw_ns, bws)
  consistent!(qx) |> wait
  get_free_dof_values(q) .= qx
  consistent!(get_free_dof_values(q)) |> wait
end

function extract_vector(du::TensorValue{2})
  VectorValue(du[1,1], du[2,2])
end

function mag(u::VectorValue{2})
  sqrt(u[1]*u[1] + u[2]*u[2])
end

function assemble_residual(du,dh,dt,F,Φ,q,q₀,dΩ,meas,g,X,f_cf,Rperp_cf,u)
  τ = 0.5
  res_u(v) = ∫( (du⋅(g⋅v))*(1/meas) )dΩ + ∫( dt*q*( (Rperp_cf⋅F)⋅v) )dΩ - ∫( dt*Φ*(∇⋅v) )dΩ
  res_p(r) = ∫(r*dh*meas )dΩ + ∫( dt*r*(∇⋅F) )dΩ
  res_d(v) = ∫( -τ*(q-q₀)*( (Rperp_cf⋅F)⋅v) )dΩ - ∫( τ*dt*(u⋅∇(q))*( (Rperp_cf⋅F)⋅v)*(1/meas) )dΩ
  dx = 1.0
  #res_d(v) = ∫(dt*dx*dx*0.5*((mag∘(u.plus)) + (mag∘(u.minus)))*mean(α)*jump((extract_vector∘∇(v))⋅n_Γ)*jump((extract_vector∘∇(F))⋅n_Γ))dΓ
  res((v,r)) = res_u(v) + res_p(r) + res_d(v)
  assemble_vector(res, X)
end

#it = iterate(solT)
#while !isnothing(it)
#  data, state = it
#  t, xh = data
#  odeopcache = state[2][5][2]
#  yh = odeopcache.diagnostics

#  uh,ph = xh
#  qh,Fh,Φh = yh

#  i_am_main(ranks) && println(t)

#  writevtk_with_cell_geomap(LatLonMapCellField(Ω),Ω,"shallow_water_sol/solT_$t.vtu",
#     cellfields=["vel"=>covariant_basis_cf⋅(1/meas*uh),"p"=>ph,"vort"=>qh],
#     append=false)

#  global it = iterate(solT, state)
#end

u, h = x
xv = get_free_dof_values(x)
dx = copy(xv)
q, F, Φ = xd
q₀ = FEFunction(H,copy(get_free_dof_values(q)))
u₀ = FEFunction(U,copy(get_free_dof_values(u)))
h₀ = FEFunction(P,copy(get_free_dof_values(h)))
qᵢ = FEFunction(H,copy(get_free_dof_values(q)))
uᵢ = FEFunction(U,copy(get_free_dof_values(u)))
hᵢ = FEFunction(P,copy(get_free_dof_values(h)))
writevtk_with_cell_geomap(LatLonMapCellField(Ω),Ω,"shallow_water_sol/solT_init",
   cellfields=["vel"=>covariant_basis_cf⋅(1/meas*u),"h"=>h,"vort"=>q],
   append=false)
for step in 1:nsteps
  get_free_dof_values(q₀) .= get_free_dof_values(q)
  get_free_dof_values(u₀) .= get_free_dof_values(u)
  get_free_dof_values(h₀) .= get_free_dof_values(h)
  for it in 1:4
    diagnostics!(q,F,Φ,0.5*(u₀+u),0.5*(h₀+h),dΩ,meas,g,ginv,U,P,H,R,cgu_ns,cgp_ns,f_cf,Rperp_cf,Mu,Mp)
    b = assemble_residual(u-u₀,h-h₀,dt,F,Φ,q,q₀,dΩ,meas,g,X_prog,f_cf,Rperp_cf,0.5*(u₀+u))
    bs = allocate_in_domain(J)
    copy!(bs, b)
    dxs = allocate_in_domain(J)
    fill!(dxs, 0.0)
    solve!(dxs, gmres_ns, bs)
    consistent!(dxs) |> wait
    dx .= dxs
    xv .= xv .- dx
    consistent!(xv) |> wait
  end
  i_am_main(ranks) && println(step)
  writevtk_with_cell_geomap(LatLonMapCellField(Ω),Ω,"shallow_water_sol/solT_$step",
     cellfields=["vel"=>covariant_basis_cf⋅(1/meas*u),"h"=>h,"vort"=>q,"F"=>covariant_basis_cf⋅(1/meas*F),"Phi"=>Φ],
     append=false)
end

qᵢ= qᵢ - q
uᵢ= uᵢ - u
hᵢ= hᵢ - h

Eq = sum(∫(qᵢ*qᵢ)dΩ)
Eu = sum(∫(uᵢ⋅uᵢ)dΩ)
Eh = sum(∫(hᵢ*hᵢ)dΩ)
if (i_am_main(ranks))
  println(Eq)
  println(Eu)
  println(Eh)
end
