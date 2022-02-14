module DarcyCubedSphereTestsSeq

using Test
using Gridap
using GridapGeosciences
using Plots
using FillArrays

include("../ConvergenceAnalysisTools.jl")
include("../DarcyCubedSphereTests.jl")

# Testing cubed sphere mesh with analytical geometric mapping
n_values=generate_n_values(2)
hs=[2.0/n for n in n_values]
model_args_series=[(n,) for n in n_values]
@time hs0,k0errors,s0=convergence_study(solve_darcy,hs,model_args_series,0,2)
@test round(s0,digits=3) ≈ 1.0

n_values=generate_n_values(2,n_max=50)
hs=[2.0/n for n in n_values]
model_args_series=[(n,) for n in n_values]
@time hs1,k1errors,s1=convergence_study(solve_darcy,hs,model_args_series,1,4)
@test round(s1,digits=3) ≈ 1.932

n_values=generate_n_values(2)
hs=[2.0/n for n in n_values]
model_args_series=zip(n_values,Fill(2,length(n_values)))

# Testing cubed sphere mesh with polynomial geometric mapping
@time hs0,k0errors,s0=convergence_study(solve_darcy,hs,model_args_series,0,2)
@test round(s0,digits=3) ≈ 1.0

n_values=generate_n_values(2,n_max=50)
hs=[2.0/n for n in n_values]
model_args_series=zip(n_values,Fill(2,length(n_values)))

@time hs1,k1errors,s1=convergence_study(solve_darcy,hs,model_args_series,1,4)
@test round(s1,digits=3) ≈ 1.936

# plot([hs0,hs1],[k0errors,k1errors],
#      xaxis=:log, yaxis=:log,
#      label=["L2 k=0","L2 k=1"],
#      shape=:auto,
#      xlabel="h",ylabel="L2 error norm")

end # module
