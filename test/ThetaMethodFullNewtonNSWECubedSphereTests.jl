module ThetaMethodFullNewtonNSWECubedSphereTests

using Test
using Gridap
using GridapGeosciences
using Plots
using LinearAlgebra
using WriteVTK
using JLD
using LineSearches

# Solves the steady state Williamson2 test case for the shallow water equations on a sphere
# of physical radius 6371220m. Involves a modified coriolis term that exactly balances
# the potential gradient term to achieve a steady state
# reference:
# D. L. Williamson, J. B. Drake, J. J.HackRüdiger Jakob, P. N.Swarztrauber, (1992)
# J Comp. Phys. 102 211-224

# Constants of the Williamson2 test case
const α  = π/4.0              # deviation of the coriolis term from zonal forcing
const U₀ = 38.61068276698372  # velocity scale
const H₀ = 2998.1154702758267 # mean fluid depth

# Modified coriolis term
function fₘ(xyz)
   θϕr   = xyz2θϕr(xyz)
   θ,ϕ,r = θϕr
   2.0*Ωₑ*( -cos(θ)*cos(ϕ)*sin(α) + sin(ϕ)*cos(α) )
end

# Initial velocity
function u₀(xyz)
  θϕr   = xyz2θϕr(xyz)
  θ,ϕ,r = θϕr
  u     = U₀*(cos(ϕ)*cos(α) + cos(θ)*sin(ϕ)*sin(α))
  v     = -U₀*sin(θ)*sin(α)
  spherical_to_cartesian_matrix(θϕr)⋅VectorValue(u,v,0)
end

# Initial fluid depth
function h₀(xyz)
  θϕr   = xyz2θϕr(xyz)
  θ,ϕ,r = θϕr
  h  = -cos(θ)*cos(ϕ)*sin(α) + sin(ϕ)*cos(α)
  H₀ - (rₑ*Ωₑ*U₀ + 0.5*U₀*U₀)*h*h/g
end

# Topography
function topography(xyz)
  0.0
end

# Compute initial volume flux
function F₀(U,V,dΩ)
  a(u,v) = ∫(v⋅u)dΩ
  b(v)   = ∫(h₀*(v⋅u₀))dΩ
  solve(AffineFEOperator(a,b,U,V))
end

# Compute initial potential vorticity
function q₀(R,S,n,dΩ)
  a(r,s) = ∫( s*h₀*r )dΩ
  b(s)   = ∫( s*fₘ - ⟂(∇(s),n)⋅u₀ )dΩ
  solve(AffineFEOperator(a,b,R,S))
end

# Generate initial monolothic solution
function uhqF₀(q₀,F₀,X,Y,dΩ)
  a((u,p,r,u2),(v,q,s,v2))=∫(v⋅u+q*p+s*r+v2⋅u2)dΩ
  b((v,q,s,v2))=∫( s*q₀ + v2⋅F₀ )dΩ
  solve(AffineFEOperator(a,b,X,Y))
end

function new_vtk_step(Ω,file,hn,un)
  createvtk(Ω,
            file,
            cellfields=["hn"=>hn, "un"=>un],
            nsubcells=4)
end

function generate_energy_plots(outdir,N,ke,pe,kin_to_pot,pot_to_kin)
  plot(1:N,
       [ke,pe],
       label=["Ek (kinetic)" "Ep (potential)"],
       xlabel="Step",ylabel="Energy")
  savefig(joinpath(outdir,"energy.png"))
  plot(1:N,
       [kin_to_pot,-pot_to_kin,kin_to_pot-pot_to_kin],
       label=["hⁿ∇⋅uⁿ" "uⁿ⋅∇(hⁿ)" "Balance"],
       xlabel="Step",ylabel="Kinetic to potential energy balance",legend = :outertopleft)
  savefig(joinpath(outdir,"kinetic_to_potential_energy_balance.png"))
end

"""
  Solves the nonlinear rotating shallow water equations
  T : [0,T] simulation interval
  N : number of time subintervals
  θ : Theta-method parameter [0,1)
"""
function solve_nswe_theta_method_full_newton(
      model,order,degree,θ,T,N;
      write_results=false,
      out_dir="nswe_ncells_$(num_cells(model))_order_$(order)_theta_method_full_newton",
      out_period=N/10)

  RT = ReferenceFE(raviart_thomas,Float64,order)
  DG = ReferenceFE(lagrangian,Float64,order)
  CG = ReferenceFE(lagrangian,Float64,order+1)

  V  = FESpace(model,RT; conformity=:Hdiv) # Velocity and mass flux FE space
  Q  = FESpace(model,DG; conformity=:L2)   # Fluid depth FE space
  S  = FESpace(model,CG; conformity=:H1)   # Potential vorticity FE space
  U  = TrialFESpace(V)
  P  = TrialFESpace(Q)
  R  = TrialFESpace(S)

  Y = MultiFieldFESpace([V,Q,S,V])         # Monolithic FE space
  X = MultiFieldFESpace([U,P,R,U])

  Ω  = Triangulation(model)
  n  = get_normal_vector(model)
  dΩ = Measure(Ω,degree)
  dω = Measure(Ω,degree,ReferenceDomain())

  un = interpolate_everywhere(u₀,U); unv=get_free_dof_values(un)
  hn = interpolate_everywhere(h₀,P); hnv=get_free_dof_values(hn)
  b  = interpolate_everywhere(topography,P); bv=get_free_dof_values(b)

  # Compute:
  #     - Initial potential vorticity (q₀)
  #     - Initial volume flux (F₀)
  #     - Initial full solution
  ΔuΔhqF=uhqF₀(q₀(R,S,n,dΩ),F₀(U,V,dΩ),X,Y,dΩ)
  Δu,Δh,_,_ = ΔuΔhqF
  function run_simulation(pvd=nothing)
    # Allocate work space vectors
    # if (write_results)
    #   ke  = Vector{Float64}(undef,N)
    #   pe  = Vector{Float64}(undef,N)
    #   kin_to_pot = Vector{Float64}(undef,N)
    #   pot_to_kin = Vector{Float64}(undef,N)
    #   mass = Vector{Float64}(undef,N)
    # end
    dt  = T/N
    τ   = dt/2 # APVM stabilization parameter
    nlcache=nothing
    for step=1:N
       e = hn-h₀;err_h = sqrt(sum(∫(e⋅e)*dΩ))
       e = un-u₀;err_u = sqrt(sum(∫(e⋅e)*dΩ))
       println("step=", step, ",\terr_u: ", err_u, ",\terr_h: ", err_h,
               " ", norm(get_free_dof_values(Δu)), " ", norm(get_free_dof_values(Δh)))

       ui(Δu)  = un       + (1-θ) * Δu
       hi(Δh)  = hn       + (1-θ) * Δh
       hbi(Δh) = hn + b   + (1-θ) * Δh

       # APVM residual
       req1((Δu,Δh,qvort,F),(v,q,s,v2)) = ∫(v⋅Δu - dt*(∇⋅(v))*(g*hbi(Δh) + 0.5*ui(Δu)⋅ui(Δu))
                                         + dt*(qvort-τ*(ui(Δu)⋅∇(qvort)))*(v⋅⟂(F,n)))dΩ
       req2((Δu,Δh,qvort,F),(v,q,s,v2)) = ∫(q*Δh)dΩ + ∫(dt*q*(DIV(F)))dω
       req3((Δu,Δh,qvort,F),(v,q,s,v2)) = ∫(s*qvort*hi(Δh) + ⟂(∇(s),n)⋅ui(Δu) - s*fₘ)dΩ
       req4((Δu,Δh,qvort,F),(v,q,s,v2)) = ∫(v2⋅F - v2⋅(hi(Δh)*ui(Δu)))dΩ
       res(x,y)                         = req1(x,y)+req2(x,y)+req3(x,y)+req4(x,y)

       # APVM jacobian
       jeq1((Δu,Δh,qvort,F),(du,dp,dr,du2),(v,q,s,v2))=∫(v⋅du +
                            +  dt*(dr    - τ*(ui(Δu)⋅∇(dr)+ui(du)⋅∇(qvort)))*(v⋅⟂(F  ,n))
                            +  dt*(qvort - τ*(             ui(Δu)⋅∇(qvort)))*(v⋅⟂(du2,n))
                            -  dt*(∇⋅(v))*(g*hbi(dp) +ui(Δu)⋅ui(du)))dΩ
       jeq2((Δu,Δh,qvort,F),(du,dp,dr,du2),(v,q,s,v2))=∫(q*dp)dΩ+∫(dt*q*(DIV(du2)))dω
       jeq3((Δu,Δh,qvort,F),(du,dp,dr,du2),(v,q,s,v2))=∫(s*qvort*hi(dr)+
                                                         s*dr*hi(Δh)+
                                                         ⟂(∇(s),n)⋅ui(du))dΩ
       jeq4((Δu,Δh,qvort,F),(du,dp,dr,du2),(v,q,s,v2))=∫(v2⋅du2-
                                                         v2⋅(hi(Δh)*ui(du))-
                                                         v2⋅(hi(dp)*ui(Δu)))dΩ

       jac(u,du,v)= jeq1(u,du,v)+jeq2(u,du,v)+jeq3(u,du,v)+jeq4(u,du,v)

       # Solve fully-coupled monolithic nonlinear problem
       # Use previous time-step solution, ΔuΔhqF, as initial guess
       # Overwrite solution into ΔuΔhqF
       op=FEOperator(res,jac,X,Y)
       nls=NLSolver(show_trace=true, method=:newton)
       solver=FESolver(nls)

       solve!(ΔuΔhqF,solver,op)

       # Update current solution
       unv .= unv .+ get_free_dof_values(Δu)
       hnv .= hnv .+ get_free_dof_values(Δh)

       if (write_results)
        #  ke[step]=Eₖ(un,H,dΩ)
        #  pe[step]=Eₚ(hn,g,dΩ)
        #  kin_to_pot[step]=compute_kin_to_pot!(u1v,unv,divvh,hnv)
        #  pot_to_kin[step]=compute_pot_to_kin!(h1v,hnv,qdivu,unv)
        #  mass[step] = compute_mass(L2MM,hnv)
         if mod(step, out_period) == 0
           println(step)
           pvd[Float64(step)] = new_vtk_step(Ω,joinpath(out_dir,"n=$(step)"),hn,un)
         end
       end
    end
    if (write_results)
      pvd[Float64(N)] = new_vtk_step(Ω,joinpath(out_dir,"n=$(N)"),hn,un)
      vtk_save(pvd)
      # generate_energy_plots(out_dir,N,ke,pe,kin_to_pot,pot_to_kin)
      # # save global scalar snapshots
      # save(joinpath(out_dir,"wave_eq_geosciences_data.jld"), "hn_dot_div_un", kin_to_pot,
      #                                           "un_dot_grad_hn", pot_to_kin,
      #                                           "mass", mass,
      #                                           "kinetic", ke,
      #                                           "potential", pe)
    end
    #un,hn
  end
  if (write_results)
    rm(out_dir,force=true,recursive=true)
    mkdir(out_dir)
    pvdfile=joinpath(out_dir,
      "nswe_ncells_$(num_cells(model))_order_$(order)_theta_method_full_newton")
    paraview_collection(run_simulation,pvdfile)
  else
    run_simulation()
  end
end

T=432000
model=CubedSphereDiscreteModel(10;radius=rₑ)
N=120
order=0
degree=2
θ=0.5
@time solve_nswe_theta_method_full_newton(model, order, degree, θ, T, N;
                                          write_results=false,out_period=10)


end # module
