module GeostrophicThermalShallowWaterExplicit

using Gridap
using GridapGeosciences

include("ThermoGeostrophicInitialConditions.jl")

# Thermogeostrophic configuration using the explicit thermal shallow water
# solver.

order  = 1
degree = 4

n      = 24
nstep  = 20*24*120 # 20 days
dt     = 30.0

model = CubedSphereDiscreteModel(n; radius=rₑ)

hf, uf = thermal_shallow_water_explicit_time_stepper(model, order, degree,
                                    h₀, u₀, S₀, f,
                                    dt, 0.5*dt, nstep;
                                    write_solution=true,
                                    write_solution_freq=240,
                                    write_diagnostics=true,
                                    write_diagnostics_freq=1,
                                    dump_diagnostics_on_screen=true)

end
