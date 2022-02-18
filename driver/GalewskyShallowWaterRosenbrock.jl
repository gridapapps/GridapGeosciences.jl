module GalewskyShallowWaterRosenbrock

using Gridap
using GridapGeosciences

include("GalewskyInitialConditions.jl")

# Solves the Galewsky test case for the shallow water equations on a sphere
# of physical radius 6371220m. Involves a shear flow instability of a zonal
# jet triggered by an initial gravity wave.
# reference:
#   Galewsky, Scott and Polvani (2004) Tellus, 56A 429-440

order  = 2
degree = 10

# magnitude of the descent direction of the implicit solve;
# neutrally stable for 0.5, L-stable for 1+sqrt(2)/2
λ = 1.0 + 0.5*sqrt(2.0)

n      = 32
dt     = 360.0
nstep  = Int(24*60^2*20/dt) # 20 days

model = CubedSphereDiscreteModel(n; radius=rₑ)

hf, uf = shallow_water_rosenbrock_time_stepper(model, order, degree,
                                               h₀, u₀, f, g, H₀,
                                               λ, dt, 180.0, nstep;
                                               write_solution=true,
                                               write_solution_freq=240,
                                               write_diagnostics=true,
                                               write_diagnostics_freq=1,
                                               dump_diagnostics_on_screen=true)

end
