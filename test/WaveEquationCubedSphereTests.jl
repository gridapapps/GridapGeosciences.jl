module WaveEquationCubedSphereTests

using Test
using Gridap
using GridapGeosciences
using Plots
using LinearAlgebra
using WriteVTK
using DataFrames
using CSV

# Initial depth
function h₀(xyz)
  x,y,z = xyz
  xᵢ,yᵢ,zᵢ = (1,0,0)
  1.0 + 0.0001*exp(-5*((xᵢ-x)^2+(yᵢ-y)^2+(zᵢ-z)^2))
end

# Initial velocity
u₀(xyz) = zero(xyz)

function new_vtk_step(Ω,file,hn,un)
  createvtk(Ω,
            file,
            cellfields=["hn"=>hn, "un"=>un],
            nsubcells=4)
end

function _ssrk2_update!(res,MM,MMchol,DIV,α,a,b)
  mul!(res, MM, a)            # res <- MM*a
  mul!(res, DIV, b, α, 1.0)   # res <- 1.0*res + α*DIV*b
  ldiv!(MMchol,res)           # res <- inv(MM)*res
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
  Solves the wave equation using a 2nd order
  Strong-Stability-Preserving Runge-Kutta
  explicit time integration scheme (SSRK2)

  g : acceleration due to gravity
  H : (constant) reference layer depth
  T : [0,T] simulation interval
  N : number of time subintervals
"""
function solve_wave_equation_ssrk2(
      model,order,degree,g,H,T,N;
      write_results=false,
      out_dir="wave_eq_ncells_$(num_cells(model))_order_$(order)_ssrk2",
      out_period=N/10)

  RT=ReferenceFE(raviart_thomas,Float64,order)
  DG=ReferenceFE(lagrangian,Float64,order)
  V = FESpace(model,RT; conformity=:Hdiv)
  Q = FESpace(model,DG; conformity=:L2)

  U = TrialFESpace(V)
  P = TrialFESpace(Q)

  Ω  = Triangulation(model)
  dΩ = Measure(Ω,degree)
  dω = Measure(Ω,degree,ReferenceDomain())

  # Build mass matrix in RT and DG spaces
  # and their sparse Cholesky factors
  amm(u,v) = ∫(v⋅u)dΩ
  RTMM=assemble_matrix(amm,U,V)
  L2MM=assemble_matrix(amm,P,Q)
  RTMMchol=lu(RTMM)
  L2MMchol=lu(L2MM)

  # Build g*DIV(v)*h and H*q*DIV(u)
  ad(h,v) = ∫(DIV(v)*h)dω
  divvh=assemble_matrix(ad,P,V)
  adt(u,q) = ∫(q*DIV(u))dω
  qdivu=assemble_matrix(adt,U,Q)

  # Interpolate initial condition into FE spaces
  hn=interpolate_everywhere(h₀,P); hnv=get_free_dof_values(hn)
  un=interpolate_everywhere(u₀,U); unv=get_free_dof_values(un)
  function run_simulation(pvd=nothing)
    # Allocate work space vectors
    h1v = similar(get_free_dof_values(hn))
    h2v = similar(h1v)
    u1v = similar(get_free_dof_values(un))
    u2v = similar(u1v)
    if (write_results)
      ke  = Vector{Float64}(undef,N)
      pe  = Vector{Float64}(undef,N)
      kin_to_pot = Vector{Float64}(undef,N)
      pot_to_kin = Vector{Float64}(undef,N)
      mass = Vector{Float64}(undef,N)
      initialize_csv(joinpath(out_dir,"wave_eq_geosciences_data.csv"),
                  "time", "hn_dot_div_un", "un_dot_grad_hn",
                  "mass", "kinetic", "potential")
    end
    dt  = T/N
    dtg = dt*g
    dtH = dt*H


    for step=1:N
       # 1st step
       # inv(L2MM)*(L2MM*hnv - dt*Hqdivu*unv)
       # inv(RTMM)*(RTMM*unv + dt*gdivvh*hnv)
       _ssrk2_update!(h1v, L2MM, L2MMchol, qdivu,-dtH, hnv, unv)
       _ssrk2_update!(u1v, RTMM, RTMMchol, divvh, dtg, unv, hnv)

       # 2nd step
       # inv(L2MM)*(L2MM*h1v - dt*Hqdivu*u1v)
       # inv(RTMM)*(RTMM*u1v + dt*gdivvh*h1v)
       _ssrk2_update!(h2v, L2MM, L2MMchol, qdivu, -dtH, h1v, u1v)
       _ssrk2_update!(u2v, RTMM, RTMMchol, divvh,  dtg, u1v, h1v)

       # Averaging steps
       hnv .= 0.5 .* ( hnv .+ h2v )
       unv .= 0.5 .* ( unv .+ u2v )

       if (write_results)
         ke[step]=Eₖ(un,H,dΩ)
         pe[step]=Eₚ(hn,hn,0.5*g,dΩ)
         kin_to_pot[step]=compute_kin_to_pot!(u1v,unv,divvh,hnv)
         pot_to_kin[step]=compute_pot_to_kin!(h1v,hnv,qdivu,unv)
         mass[step] = compute_total_mass!(h1v,L2MM,hnv)
         if mod(step, out_period) == 0
           println(step)
           pvd[dt*Float64(step)] = new_vtk_step(Ω,joinpath(out_dir,"n=$(step)"),hn,un)
         end
         # save global scalar snapshots
         append_to_csv(joinpath(out_dir,"wave_eq_geosciences_data.csv");
                        time = step*dt,
                        hn_dot_div_un = kin_to_pot[step],
                        un_dot_grad_hn = pot_to_kin[step],
                        mass = mass[step],
                        kinetic = ke[step],
                        potential= pe[step])
       end
    end
    if (write_results)
      pvd[dt*Float64(N)] = new_vtk_step(Ω,joinpath(out_dir,"n=$(N)"),hn,un)
      vtk_save(pvd)
      generate_energy_plots(out_dir,N,ke,pe,kin_to_pot,pot_to_kin)
    end
    un,hn
  end
  if (write_results)
    rm(out_dir,force=true,recursive=true)
    mkdir(out_dir)
    pvdfile=joinpath(out_dir,"wave_eq_ncells_$(num_cells(model))_order_$(order)_ssrk2")
    paraview_collection(run_simulation,pvdfile)
  else
    run_simulation()
  end
end

model=CubedSphereDiscreteModel(12)
g=1.0
H=1.0
T=2π
N=2000
order=0
degree=4
@time un,hn =
  solve_wave_equation_ssrk2(model,order,degree,g,H,T,N;write_results=false,out_period=10)


@test Eₖ(un,H,Measure(Triangulation(model),degree)) ≈ 1.6984501177784049e-10


end # module
