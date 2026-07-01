using Gridap
using Gridap.Geometry, Gridap.CellData, Gridap.Arrays, Gridap.FESpaces
using Gridap.ReferenceFEs, Gridap.Fields, Gridap.Geometry
using GridapGeosciences

include("helper_funcs.jl")
include("ambient_funcs.jl")
include("panel_funcs.jl")


################################################################################
#### Run benchmark
################################################################################
degree = 10
orders = collect(1:3)
for order in orders
  benchmark_ambient(order,degree)
  benchmark_reference_panel(order,degree)
end

@test true

return_output = false
# if return_output=true, save output, and plot.
# WARNING! follow intructions in readme about packages to add, and run the below
# code locally. Also uncomment lines in benchmark_ambient and benchmark_reference_panel
# using GFlops
# using DrWatson
# using BenchmarkTools
if return_output

  dir = datadir("gradgrad_degree$(degree)")
  orders = collect(1:3)
  for order in orders
    benchmark_ambient(order,degree,dir,return_output)
    benchmark_reference_panel(order,degree,dir,return_output)
  end
  ##############################################################################
  #### Collect results
  ##############################################################################
  using DataFrames
  df = collect_results(dir)

  ambient = df[df.model.=="ambient",:]
  panel = df[df.model.=="panel",:]
  # orders = ambient[:,:order]

  data = [:ops,:t,:flops]
  for (i,sym) in enumerate(data)
  println("Ratio $(sym): ", ambient[:,sym]./panel[:,sym])
  end

  ##############################################################################
  #### Plot results
  ##############################################################################
  using Plots
  markers= [:circle :rect  :diamond ]
  markersize = [6 7 6]
  colors = palette(:tab10)
  default(; fontfamily="Computer Modern");

  data = [:ops,:t,:flops]
  labs = ["Operations", "Time", "Flops"]

  plot()
  for (i,(sym,lab)) in enumerate(zip(data,labs))
    plot!(orders, ambient[:,sym]./panel[:,sym],
          lw=2,marker=markers[i],ms=markersize[i],color=colors[i],
          label=lab )
  end
  plot!(show=true)
  plot!(shape=:auto,
        xlabel="Order",
        ylabel="Extrinsic/Instrinsic",
        xtickfontsize=11,ytickfontsize=11,
        xguidefontsize=12,yguidefontsize=12,
        legendfontsize=10,
        legend=:bottomleft,
        framestyle = :box,
        ylimits=(0,10),
        xticks = (orders, orders)
        )
  savefig(plotsdir("benchmark.pdf"))
end
