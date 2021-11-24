using Pkg
Pkg.add("PackageCompiler")
using PackageCompiler

pkgs = Symbol[]
push!(pkgs, :GridapGeosciences)

if VERSION >= v"1.4"
    append!(pkgs, [Symbol(v.name) for v in values(Pkg.dependencies()) if v.is_direct_dep],)
else
    append!(pkgs, [Symbol(name) for name in keys(Pkg.installed())])
end

pkgs = [:Gridap,:GridapDistributed]
create_sysimage(pkgs,
  sysimage_path=joinpath(@__DIR__,"GridapGeosciences.so"),
  precompile_execution_file=joinpath(@__DIR__,"..","mpi/Williamson2ThetaMethodFullNewtonTests.jl"))
