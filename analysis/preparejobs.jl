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


dicts_cartesian=generate_2d_dicts(:cartesian,:gamg,collect(3:8),[16,32,64,128,256,512])
dicts=generate_2d_dicts(:p4est,:gamg,collect(3:8),collect(4:9))
append!(dicts,dicts_cartesian)
template = read(projectdir("jobtemplate.sh"),String)
for params in dicts
   fparams=convert_nc_np_to_prod(params)
   jobfile = datadir(jobname(fparams,"sh"))
   open(jobfile,"w") do io
     render(io,template,jobdict(params))
   end
end
