module GeostrophicThermalShallowWaterExplicit

using Gridap
using GridapGeosciences

include("ThermoGeostrophicInitialConditions.jl")

# Thermogeostrophic configuration using the explicit thermal shallow water
# solver.

order  = 1
degree = 4

n      = 24
nstep  = 20*24*30 # 20 days
dt     = 120.0

model = CubedSphereDiscreteModel(n; radius=rₑ)

hf, uf = thermal_shallow_water_mat_adv_explicit_time_stepper(model, order, degree,
                                    h₀, u₀, s₀, f,
                                    dt, 0.5*dt, 0.0*dt, nstep;
                                    write_solution=true,
                                    write_solution_freq=360,
                                    write_diagnostics=true,
                                    write_diagnostics_freq=1,
                                    dump_diagnostics_on_screen=true)

end
