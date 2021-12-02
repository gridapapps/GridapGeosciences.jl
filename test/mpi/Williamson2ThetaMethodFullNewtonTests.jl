module Williamsom2ThetaMethodFullNewtonTestsMPI

using  PartitionedArrays
using  Test
using  FillArrays
using  Gridap
using  GridapDistributed
using  GridapGeosciences
using  GridapPETSc
using SparseArrays
using MPI

const PArrays=PartitionedArrays

Base.unaliascopy(A::Gridap.Arrays.SubVector) = typeof(A)(Base.unaliascopy(A.vector), A.pini, A.pend)

include("../sequential/Williamson2InitialConditions.jl")

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
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  14, 1000)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 28, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 29, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetCntl(mumpsmat[], 3, 1.0e-6)
end


function set_ksp_mm(ksp)
  @check_error_code GridapPETSc.PETSC.KSPSetOptionsPrefix(ksp[],"mm_")
  @check_error_code GridapPETSc.PETSC.KSPSetFromOptions(ksp[])
  #@check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
end

function main(parts)
  GridapPETSc.with(args=split(options)) do
      if (PArrays.get_part_id(parts)==1)
        println(options)
      end

      # Solves the steady state Williamson2 test case for the shallow water equations on a sphere
      # of physical radius 6371220m. Involves a modified coriolis term that exactly balances
      # the potential gradient term to achieve a steady state
      # reference:
      # D. L. Williamson, J. B. Drake, J. J.HackRüdiger Jakob, P. N.Swarztrauber, (1992)
      # J Comp. Phys. 102 211-224

      l2_err_u = [0.011370921987771046 , 0.002991229234176333 ]
      l2_err_h = [0.005606685579166809, 0.001458107077200681 ]

      order=1
      degree=4
      θ=0.5
      for i in 1:1
        n      = i+1
        nstep  = 5*2^n
        Uc     = sqrt(g*H₀)
        dx     = 2.0*π*rₑ/(4*n)
        dt     = 0.25*dx/Uc
        println("timestep: ", dt)   # gravity wave time step
        T      = dt*nstep
        τ      = dt/2
        model  = CubedSphereDiscreteModel(parts, n; radius=rₑ)
        nls    = PETScNonlinearSolver(mysnessetup)
        mmls   = PETScLinearSolver(set_ksp_mm)
        hf, uf, _ = shallow_water_theta_method_full_newton_time_stepper(nls, model, order, degree,
                                                            h₀, u₀, f₀, topography,
                                                            g, θ, T, nstep, τ;
                                                            mass_matrix_solver=mmls,
                                                            am_i_root=PArrays.get_part_id(parts)==1,
                                                            write_solution=false,
                                                            write_solution_freq=5,
                                                            write_diagnostics=true,
                                                            write_diagnostics_freq=1,
                                                            dump_diagnostics_on_screen=true)

        Ω     = Triangulation(model)
        dΩ    = Measure(Ω, degree)
        hc    = CellField(h₀, Ω)
        e     = h₀-hf
        err_h = sqrt(sum(∫(e⋅e)*dΩ))/sqrt(sum(∫(hc⋅hc)*dΩ))
        uc    = CellField(u₀, Ω)
        e     = u₀-uf
        err_u = sqrt(sum(∫(e⋅e)*dΩ))/sqrt(sum(∫(uc⋅uc)*dΩ))
        if PArrays.get_part_id(parts)==1
          println("n=", n, ",\terr_u: ", err_u, ",\terr_h: ", err_h)
        end
        #@test abs(err_u - l2_err_u[i]) < 10.0^-12
        #@test abs(err_h - l2_err_h[i]) < 10.0^-12
      end
  end
end
MPI.Init()
prun(main,mpi,MPI.Comm_size(MPI.COMM_WORLD))

end # module
