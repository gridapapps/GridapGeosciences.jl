using Plots
using DrWatson
using DataFrames
using CSV

fn = plotsdir("summary_1.csv")
df = CSV.File(fn) |> DataFrame

n=[9437186,37748738]

df_n1=df[df.ngdofs .== n[1], :]
df_n2=df[df.ngdofs .== n[2], :][1:5,:]

x1 = df_n1[:,:np]
y1 = df_n1[:,:Simulation]

x2 = df_n2[:,:np]
y2 = df_n2[:,:Simulation]

function doplot()
  plt = plot(x1,y1,
     thickness_scaling = 1.2,
     xaxis=("Number of cores",:log),
     yaxis=("Wall time [s]",:log),
     xticks=([96,192,384,672,960],
     ["96","192","384","672","960"]),
     yticks=([66,78,87,149,185,200,253,280,336,377],
     ["66","78","87","149","185","200","253","280","336","377"]),
     minorgrid=true,
     title="Galewsky. 10x steps (θ=0.5,dt=480[s]). Newton/GMRES-RAS.",
     titlefont=font(8),
     shape=:square,
     label="256x256x6 [9.4MDoFs]",
     legend=:bottomleft,
     legendfont=font(6),
     markerstrokecolor=:white,
    )

  plot!(x2,y2,
    thickness_scaling = 1.2,
    xaxis=("Number of cores",:log),
    yaxis=("Wall time [s]",:log),
    xticks=([96,192,384,672,960],
    ["96","192","384","672","960"]),
    yticks=([66,78,87,149,185,200,253,280,336,377],
    ["66","78","87","149","185","200","253","280","336","377"]),
    minorgrid=true,
    title="Galewsky. 10x time steps (θ=0.5,dt=480[s]). Newton/GMRES-RAS.",
    titlefont=font(8),
    shape=:square,
    label="512x512x6 [37.7MDoFs]",
    legend=:bottomleft,
    legendfont=font(6),
    color=:green,
    markerstrokecolor=:white,
   )

  function plot_vert(plt,x,y,ngdofs,l,p,color)
     x1 = first(x)
     y1 = first(y)
     s = y1.*(x1./x)
     println(x)
     println(s)
     plot!(x,s,xaxis=:log,yaxis=:log, color=color, label="Ideal"),
     Plots.vline!([l],
           xaxis=:log,
           yaxis=:log,
           linestyle=:dot,seriestype = :vline,
           color=color,
           label="$(p)KDoFs/core")
     plt
  end
  plot_vert(plt,x1,y1,first(df_n1.ngdofs),first(df_n1.ngdofs)/25e3,25,:blue)
  plot_vert(plt,x2,y2,first(df_n2.ngdofs),first(df_n2.ngdofs)/50e3,50,:green)
  plt
end



#unicodeplots()
#display(doplot())

gr()
doplot()
savefig(plotsdir("total_scaling.pdf"))
savefig(plotsdir("total_scaling.png"))
