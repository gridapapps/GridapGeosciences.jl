module Williamson5ShallowWaterRosenbrock

  using LinearMaps
  using SparseMatricesCSR
  using PartitionedArrays
  using Test
  using FillArrays
  using Gridap
  using GridapPETSc
  using GridapGeosciences
  using GridapDistributed
  using GridapSolvers
  using GridapP4est

  include("../sequential/Williamson2InitialConditions.jl")

  order  = 0
  degree = 4

  # magnitude of the descent direction of the implicit solve;
  # neutrally stable for 0.5, L-stable for 1+sqrt(2)/2
  λ = 1.0 + 0.5*sqrt(2.0)

  dt     = 30.0
  τ      = 0.5*dt
  nstep  = 10#Int(24*60^2*20/dt) # 20 days

  function main(distribute,parts)
    ranks = distribute(LinearIndices((prod(parts),)))

    # Change directory to the location of the script, where the mesh data files are located 
    cd(@__DIR__)

    coarse_model, cell_panels, coarse_cell_wise_vertex_coordinates = parse_cubed_sphere_coarse_model("C12-regular/connectivity-gridapgeo.txt",
                                                                                                     "C12-regular/geometry-gridapgeo.txt")

    num_uniform_refinements=0

    model = CubedSphereDiscreteModel(ranks,
                                     coarse_model,
                                     coarse_cell_wise_vertex_coordinates,
                                     cell_panels,
                                     num_uniform_refinements;
                                     radius=Rₑ,
                                     #adaptive=true,
                                     adaptive=false,
                                     order=1)

    P          = JacobiLinearSolver()
    mm_solver  = GridapSolvers.CGSolver(P;rtol=1.e-6)
    jac_solver = GMRESSolver(100;Pr=nothing,Pl=nothing,maxiter=2000,atol=1e-12,rtol=1.e-6,restart=true,m_add=20,verbose=false,name="GMRES")

    hf, uf = shallow_water_rosenbrock_time_stepper(model, order, degree,
                                                   h₀, u₀, Ωₑ, gₑ, H₀,
                                                   λ, dt, τ, nstep,
                                                   mm_solver, jac_solver;
                                                   leap_frog=true,
                                                   write_solution=true,
                                                   write_solution_freq=1,
                                                   write_diagnostics=true,
                                                   write_diagnostics_freq=1,
                                                   dump_diagnostics_on_screen=true)
  end

  with_mpi() do distribute 
    main(distribute,1)
  end
end # module
