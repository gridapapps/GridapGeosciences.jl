using Plots
using DrWatson
using DataFrames
using CSV

function doplot(file,ls,mesh)
  df = CSV.File(file) |> DataFrame
  lsrows=df[.&(df[!,:ls].==Float64(ls),(df[!,:mesh].==mesh)),:]
  x=lsrows[:,:nparts]
  ymod=lsrows[:,:Model]
  yfes=lsrows[:,:FESpaces]
  yope=lsrows[:,:AffineFEOperator]
  ysol=lsrows[:,:Solve]
  ynrm=lsrows[:,:L2Norm]
  m=maximum([maximum(ymod),maximum(yfes),maximum(yope),maximum(ynrm)])
  plt = plot(x,ymod,
             thickness_scaling = 1.2,
             xaxis=("Number of cores"),
             yaxis=("Wall time [s]"),
             ylims = (0,1.2*m),
             title="Gridap 0.2.0 LS=$(ls)",
             shape=:auto,
             label="Model ($(mesh))",
             legend=:outertopright,
             markerstrokecolor=:white,
           )
   plot!(x,yfes,label="FESpace",shape=:auto)
   plot!(x,yope,label="FEOperator",shape=:auto)
   plot!(x,ysol,label="Solver",shape=:auto)
   plot!(x,ynrm,label="L2Norm",shape=:auto)
end

function saveplot3D(file,ls,mesh)
  gr()
  doplot(file,ls,mesh)
  savefig("weak_scaling_0.2.0_3D_$(ls)_$(mesh).pdf")
end

meshes = ["cartesian"]
ls_lst_cartesian  = [2^i for i=2:5]
for mesh in meshes
  if mesh == "cartesian"
    ls_lst=ls_lst_cartesian
    file="summary.csv"
  elseif mesh == "p4est"
    #ls_lst=ls_lst_p4est
    #file="report_2D_p4est.txt"
  end
  for ls in ls_lst
    saveplot3D(plotsdir(file),ls,mesh)
  end
end
