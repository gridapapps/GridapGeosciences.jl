using Mustache
using DrWatson

jobname(args...) = replace(savename(args...;connector="_"),"="=>"_")
driverdir(args...) = normpath(projectdir("..",args...))

function jobdict(params)
  np = params[:np]
  dt = params[:dt]
  write_solution = params[:write_solution]
  write_solution_freq = params[:write_solution_freq]
  k = params[:k]
  degree = params[:degree]
  mumps_relaxation = params[:mumps_relaxation]
  Dict(
  "q" => "normal",
  "o" => datadir(jobname(params,"o.txt")),
  "e" => datadir(jobname(params,"e.txt")),
  "walltime" => "24:00:00",
  "ncpus" => np,
  "mem" => "$(prod(np)*4)gb",
  "name" => jobname(params),
  "numrefs" => params[:numrefs],
  "n" => np,
  "dt" => dt,
  "write_solution" => write_solution,
  "write_solution_freq" => write_solution_freq,
  "k" => k,
  "degree" => degree,
  "mumps_relaxation" => mumps_relaxation,
  "projectdir" => driverdir(),
  "modules" => driverdir("modules.sh"),
  "title" => datadir(jobname(params)),
  "sysimage" => driverdir("GridapGeosciences.so")
  )
end


allparams = Dict(
 :np => 8,
 :numrefs => 2,
 :write_solution => false,
 :write_solution_freq => 4,
 :dt => 480,
 :k => 1,
 :degree => 4,
 :mumps_relaxation => 1000,
 )

template = read(projectdir("jobtemplate.sh"),String)

dicts = dict_list(allparams)

for params in dicts
  jobfile = datadir(jobname(params,"sh"))
  open(jobfile,"w") do io
    render(io,template,jobdict(params))
  end
end
