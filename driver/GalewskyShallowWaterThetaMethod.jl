module GalewskyShallowWaterThetaMethod

using Gridap
using GridapGeosciences
using GridapPardiso
using SparseMatricesCSR


include("GalewskyInitialConditions.jl")

# Solves the Galewsky test case for the shallow water equations on a sphere
# of physical radius 6371220m. Involves a shear flow instability of a zonal
# jet triggered by an initial gravity wave.
# reference:
#   Galewsky, Scott and Polvani (2004) Tellus, 56A 429-440

function Base.copy(a::SparseMatrixCSR{Bi}) where Bi
  SparseMatrixCSR{Bi}(a.m,a.n,copy(a.rowptr),copy(a.colval),copy(a.nzval))
end

order  = 1
degree = 4

n      = 48
dt     = 480.0
nstep  = Int(24*60^2*20/dt) # 20 days
T      = dt*nstep
θ      = 0.5

model = CubedSphereDiscreteModel(n; radius=rₑ)

linear_solver=PardisoSolver(GridapPardiso.MTYPE_REAL_NON_SYMMETRIC,
                            GridapPardiso.new_iparm(),
                            GridapPardiso.MSGLVL_VERBOSE,
                            GridapPardiso.new_pardiso_handle())

shallow_water_theta_method_full_newton_time_stepper(model, order, degree,
                                               h₀, u₀, f, topography, g, θ, T, nstep, dt/2;
                                               linear_solver=linear_solver,
                                               sparse_matrix_type=SparseMatrixCSR{1,Float64,Int},
                                               write_solution=true,
                                               write_solution_freq=45,
                                               write_diagnostics=true,
                                               write_diagnostics_freq=1,
                                               dump_diagnostics_on_screen=true)

end
