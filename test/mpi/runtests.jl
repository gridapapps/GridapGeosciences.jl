module MPITests

using MPI
using Test

#Sysimage
sysimage=nothing
if length(ARGS)==1
   @assert isfile(ARGS[1]) "$(ARGS[1]) must be a valid Julia sysimage file"
   sysimage=ARGS[1]
end

mpidir = @__DIR__
testdir = mpidir #joinpath(mpidir,"..")
repodir = joinpath(mpidir,"../../")

function run_driver(procs,file,sysimage)
  mpiexec() do cmd
    if sysimage!=nothing
       extra_args="-J$(sysimage)"
       cmd=`$cmd -n $procs $(Base.julia_cmd()) $(extra_args) --project=$repodir $(joinpath(mpidir,file))`
    else
       cmd=`$cmd -n $procs $(Base.julia_cmd()) --project=$repodir $(joinpath(mpidir,file))`
    end
    println(cmd)
    run(cmd)
    @test true
  end
end
run_driver(4,"CubedSphereDiscreteModelsTests.jl",sysimage)

end # module
