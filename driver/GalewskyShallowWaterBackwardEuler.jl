module GalewskyShallowWaterRosenbrock

using Gridap
using GridapGeosciences

include("GalewskyInitialConditions.jl")

# Solves the Galewsky test case for the shallow water equations on a sphere
# of physical radius 6371220m. Involves a shear flow instability of a zonal
# jet triggered by an initial gravity wave.
# reference:
#   Galewsky, Scott and Polvani (2004) Tellus, 56A 429-440

order  = 1
degree = 4

n      = 48
nstep  = 20*180 # 20 days
dt     = 480.0
T      = dt*nstep
θ      = 0.0

model = CubedSphereDiscreteModel(n; radius=rₑ)

shallow_water_theta_method_full_newton_time_stepper(model, order, degree,
                                               h₀, u₀, f, g, θ, T, nstep;
                                               write_solution=true,
                                               write_solution_freq=45,
                                               write_diagnostics=true,
                                               write_diagnostics_freq=1,
                                               dump_diagnostics_on_screen=true)

end
