module GalewskyShallowWaterThetaMethod

using Gridap
using GridapDistributed
using GridapPETSc
using GridapP4est
using PartitionedArrays
const PArrays=PartitionedArrays
using GridapGeosciences
using SparseMatricesCSR
using FileIO



gamg = """
       -ksp_type gmres -ksp_rtol 1.0e-06 -ksp_atol 0.0 -ksp_gmres_restart 100
       -ksp_monitor -pc_type gamg -pc_gamg_type agg
       -mg_levels_esteig_ksp_type gmres -mg_coarse_sub_pc_type lu
       -mg_coarse_sub_pc_factor_mat_ordering_type nd -pc_gamg_process_eq_limit 50
       -pc_gamg_square_graph 9 pc_gamg_agg_nsmooths 1
       """

mumps = """
        -ksp_type preonly -ksp_error_if_not_converged true
        -pc_type lu -pc_factor_mat_solver_type mumps
        -mat_mumps_icntl_1 4
        -mat_mumps_icntl_14 100
        -mat_mumps_icntl_28 1
        -mat_mumps_icntl_29 2
        -mat_mumps_cntl_3 1.0e-6
        """

options = """
          -snes_type newtonls
          -snes_linesearch_type basic
          -snes_linesearch_damping 1.0
          -snes_rtol 1.0e-8
          -snes_atol 0.0
          -snes_monitor
          -snes_converged_reason
          -mm_ksp_type cg
          -mm_ksp_monitor
          -mm_ksp_rtol 1.0e-14
          -mm_pc_type jacobi
          """


  function main_galewsky(parts,ir,
                        np,numrefs,dt,τ,
                        write_solution,write_solution_freq,title,order,degree,
                        verbose,mumps_relaxation,nstep)

    function mysnessetup(snes)
      ksp      = Ref{GridapPETSc.PETSC.KSP}()
      pc       = Ref{GridapPETSc.PETSC.PC}()
      mumpsmat = Ref{GridapPETSc.PETSC.Mat}()
      @check_error_code GridapPETSc.PETSC.SNESSetFromOptions(snes[])
      @check_error_code GridapPETSc.PETSC.SNESGetKSP(snes[],ksp)
      @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
      @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPPREONLY)
      @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
      @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCLU)
      @check_error_code GridapPETSc.PETSC.PCFactorSetMatSolverType(pc[],GridapPETSc.PETSC.MATSOLVERMUMPS)
      @check_error_code GridapPETSc.PETSC.PCFactorSetUpMatSolverType(pc[])
      @check_error_code GridapPETSc.PETSC.PCFactorGetMatrix(pc[],mumpsmat)
      @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  4, 1)
      # percentage increase in the estimated working space
      @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  14, mumps_relaxation)
      @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 28, 2)
      @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 29, 2)
      @check_error_code GridapPETSc.PETSC.MatMumpsSetCntl(mumpsmat[], 3, 1.0e-6)
    end


    function set_ksp_mm(ksp)
      @check_error_code GridapPETSc.PETSC.KSPSetOptionsPrefix(ksp[],"mm_")
      @check_error_code GridapPETSc.PETSC.KSPSetFromOptions(ksp[])
    end


    t = PArrays.PTimer(parts,verbose=true)
    PArrays.tic!(t,barrier=true)
    ngcells, ngdofs = galewsky(parts,numrefs,dt,τ,
                              write_solution,write_solution_freq,title,order,degree,
                              verbose,mumps_relaxation,mysnessetup,set_ksp_mm,nstep)
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
      out["ir"] = ir
      save("$title.bson",out)
    end
  end

  ########

  function main(;
    np::Integer,
    nr::Integer,
    numrefs::Integer,
    dt::Float64,
    τ::Float64=dt/4,
    write_solution=false,
    write_solution_freq=4,
    title::AbstractString,
    k::Integer=1,
    degree::Integer=4,
    verbose::Bool=true,
    mumps_relaxation=1000,
    nstep::Integer=Int(24*60^2*20/dt)) # 20 days

    numrefs>=1 || throw(ArgumentError("numrefs should be larger or equal than 1"))

    prun(mpi,np) do parts
      for ir=1:nr
        GridapPETSc.with(args=split(options)) do
          str_r   = lpad(ir,ceil(Int,log10(nr)),'0')
          title_r = "$(title)_ir$(str_r)"
          main_galewsky(parts,ir,np,numrefs,dt,τ,
              write_solution,write_solution_freq,title_r,
              k,degree,
              verbose,mumps_relaxation,nstep)
        end
      end
    end
  end

  include("../../../sequential/GalewskyInitialConditions.jl")

  function galewsky(parts,numrefs,dt,τ,
                    write_solution,write_solution_freq,title,order,degree,
                    verbose,mumps_relaxation,mysnessetup,set_ksp_mm,nstep)
      T      = dt*nstep
      θ      = 0.5
      model  = CubedSphereDiscreteModel(parts, numrefs; radius=rₑ)

      nls    = PETScNonlinearSolver(mysnessetup)
      mmls   = PETScLinearSolver(set_ksp_mm)

      function ts()
        shallow_water_theta_method_full_newton_time_stepper(nls, model, order, degree,
        h₀, u₀, f, topography, g, θ, T, nstep, τ;
        mass_matrix_solver=mmls,
        am_i_root=PArrays.get_part_id(parts)==1,
        write_solution=write_solution,
        write_solution_freq=write_solution_freq,
        write_diagnostics=true,
        write_diagnostics_freq=1,
        dump_diagnostics_on_screen=true,
        output_dir=title)
      end

      if (PArrays.get_part_id(parts)==1)
        @time _,_,ndofs = ts()
      else
        _,_,ndofs = ts()
      end

      num_cells(model), ndofs
  end
end
