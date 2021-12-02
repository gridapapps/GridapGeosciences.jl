module GalewskyShallowWaterThetaMethod

using Gridap
using GridapDistributed
using GridapPETSc
using GridapP4est
using PartitionedArrays
const PArrays=PartitionedArrays
using GridapGeosciences
using SparseMatricesCSR

  function main_galewsky(parts,
                        np,numrefs,dt,τ,
                        write_solution,write_solution_freq,title,order,degree,
                        verbose,mumps_relaxation)

    t = PArrays.PTimer(parts,verbose=true)
    PArrays.tic!(t,barrier=true)
    ngcells, ngdofs = galewsky(parts,numrefs,dt,τ,
                              write_solution,write_solution_freq,title,order,degree,
                              verbose,mumps_relaxation)
    PArrays.toc!(t,"Simulation")
    display(t)
    map_main(t.data) do data
      out = Dict{String,Any}()
      merge!(out,data)
      out["ngdofs"] = ngdofs
      out["ngcells"] = ngcells
      out["np"] = np
      out["order"] = order
      out["dt"] = dt
      out["τ"] = τ
      out["degree"] = degree
      save("$title.bson",out)
    end
  end

  ########

  function main(;
    np::Integer,
    numrefs::Integer,
    dt::Float64,
    τ::Float64=dt/4,
    write_solution=false,
    write_solution_freq=4,
    title::AbstractString,
    k::Integer=1,
    degree::Integer=4,
    verbose::Bool=true,
    mumps_relaxation=100)

    numrefs>=1 || throw(ArgumentError("numrefs should be larger or equal than 1"))

    prun(mpi,np) do parts
        GridapPETSc.with(args=split(options)) do
          main(parts,np,numrefs,dt,τ,
              write_solution,write_solution_freq,title,
              k,degree,
              verbose,mumps_relaxation)
        end
    end
  end

  include("../sequential/GalewskyInitialConditions.jl")

  function galewsky(parts,numrefs,dt,τ,
                    write_solution,write_solution_freq,title,order,degree,
                    verbose,mumps_relaxation)

      nstep  = Int(24*60^2*20/dt) # 20 days
      T      = dt*nstep
      θ      = 0.5
      model  = CubedSphereDiscreteModel(parts, numrefs; radius=rₑ)
      _,_,ndofs=shallow_water_theta_method_full_newton_time_stepper(model, order, degree,
                                                    h₀, u₀, f, topography, g, θ, T, nstep, τ;
                                                    linear_solver=linear_solver,
                                                    sparse_matrix_type=SparseMatrixCSR{1,Float64,Int},
                                                    write_solution=write_solution,
                                                    write_solution_freq=write_solution_freq,
                                                    write_diagnostics=true,
                                                    write_diagnostics_freq=1,
                                                    dump_diagnostics_on_screen=true,
                                                    output_dir=title)

      num_cells(model), ndofs
  end
end
