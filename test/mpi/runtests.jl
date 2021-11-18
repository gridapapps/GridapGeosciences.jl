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
       cmd=`$cmd -n $procs --allow-run-as-root --oversubscribe $(Base.julia_cmd()) --project=$repodir $(joinpath(mpidir,file))`
    else
       cmd=`$cmd -n $procs --allow-run-as-root --oversubscribe $(Base.julia_cmd()) --project=$repodir $(joinpath(mpidir,file))`
    end
    println(cmd)
    run(cmd)
    @test true
  end
end
#run_driver(4,"CubedSphereDiscreteModelsTests.jl",sysimage)
#run_driver(4,"DarcyCubedSphereTests.jl",sysimage)
#run_driver(4,"LaplaceBeltramiCubedSphereTests.jl",sysimage)
run_driver(4,"WeakDivPerpTests.jl",sysimage)

end # module
