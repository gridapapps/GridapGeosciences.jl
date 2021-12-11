using Mustache
using DrWatson

jobname(args...) = replace(savename(args...;connector="_"),"="=>"_")
driverdir(args...) = normpath(projectdir("..",args...))

function jobdict(params)
  nr = params[:nr]
  nnodes = params[:nnodes]
  tpn = params[:tpn]
  dt = params[:dt]
  write_solution = params[:write_solution]
  write_solution_freq = params[:write_solution_freq]
  k = params[:k]
  degree = params[:degree]
  mrelax = params[:mrelax]
  nstep = params[:nstep]
  d=Dict(
  "q" => "normal",
  "o" => datadir(jobname(params,"o.txt")),
  "e" => datadir(jobname(params,"e.txt")),
  "walltime" => "04:00:00",
  "ncpus" => nnodes*48,
  "tpn" => tpn,
  "mem" => "$(nnodes*48*4)gb",
  "name" => jobname(params),
  "numrefs" => params[:numrefs],
  "n" => nnodes*tpn,
  "nthreads" => params[:nthreads],
  "nnodes" => nnodes,
  "dt" => dt,
  "ws" => write_solution,
  "write_solution_freq" => write_solution_freq,
  "k" => k,
  "nr" => nr,
  "degree" => degree,
  "mrelax" => mrelax,
  "nstep" => nstep,
  "projectdir" => driverdir(),
  "modules" => driverdir("modules.sh"),
  "title" => datadir(jobname(params)),
  "sysimage" => driverdir("GalewskyShallowWaterThetaMethod.so"),
  )
  println(d)
  d
end

allparams = Dict(
 :nnodes => [1],
 :numrefs => 5,
 :write_solution => false,
 :write_solution_freq => 4,
 :dt => 480.0,
 :k => 1,
 :degree => 4,
 :mrelax => 50000,
 :nstep => 10,
 :nr => 5,
 :nthreads => 24,
 :tpn=>2,
 )

template = read(projectdir("jobtemplate.sh"),String)

dicts = dict_list(allparams)

for params in dicts
  jobfile = datadir(jobname(params,"sh"))
  open(jobfile,"w") do io
    render(io,template,jobdict(params))
  end
end
