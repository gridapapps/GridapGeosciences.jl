using PackageCompiler
create_sysimage(:GalewskyShallowWaterThetaMethod,
  sysimage_path=joinpath(@__DIR__,"..","GalewskyShallowWaterThetaMethod.so"),
  precompile_execution_file=joinpath(@__DIR__,"warmup.jl"))
