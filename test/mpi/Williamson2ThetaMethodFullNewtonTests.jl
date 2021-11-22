module Williamsom2ThetaMethodFullNewtonTestsMPI

using  PartitionedArrays
using  Test
using  FillArrays
using  Gridap
using  GridapDistributed
using  GridapGeosciences
using  GridapPETSc

Base.unaliascopy(A::Gridap.Arrays.SubVector) = typeof(A)(Base.unaliascopy(A.vector), A.pini, A.pend)

include("../sequential/Williamson2InitialConditions.jl")

options = """
          -snes_type newtonls
          -snes_linesearch_type basic
          -snes_linesearch_damping 1.0
          -snes_rtol 1.0e-8
          -snes_atol 0.0
          -snes_monitor
          -ksp_type preonly
          -ksp_error_if_not_converged true
          -pc_type lu
          -pc_factor_mat_solver_type mumps
          -snes_converged_reason
          -mm_ksp_type cg
          -mm_ksp_monitor
          -mm_ksp_rtol 1.0e-6
          -mm_pc_type jacobi
          """

function set_ksp_mm(ksp)
  @check_error_code GridapPETSc.PETSC.KSPSetOptionsPrefix(ksp[],"mm_")
  @check_error_code GridapPETSc.PETSC.KSPSetFromOptions(ksp[])
  @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
end

function main(parts)
  GridapPETSc.with(args=split(options)) do
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
      for i in 1:2
        n      = i+1
        nstep  = 5*n
        Uc     = sqrt(g*H₀)
        dx     = 2.0*π*rₑ/(4*n)
        dt     = 0.25*dx/Uc
        println("timestep: ", dt)   # gravity wave time step
        T      = dt*nstep
        τ      = dt/2
        model  = CubedSphereDiscreteModel(parts, n; radius=rₑ)
        nls    = PETScNonlinearSolver()
        mmls   = PETScLinearSolver(set_ksp_mm,parts.comm)
        #hf, uf =
        shallow_water_theta_method_full_newton_time_stepper(nls, model, order, degree,
                                                            h₀, u₀, f₀, topography,
                                                            g, θ, T, nstep, τ;
                                                            mass_matrix_solver=mmls,
                                                            write_solution=false,
                                                            write_solution_freq=5,
                                                            write_diagnostics=true,
                                                            write_diagnostics_freq=1,
                                                            dump_diagnostics_on_screen=true)

        # Ω     = Triangulation(model)
        # dΩ    = Measure(Ω, degree)
        # hc    = CellField(h₀, Ω)
        # e     = h₀-hf
        # err_h = sqrt(sum(∫(e⋅e)*dΩ))/sqrt(sum(∫(hc⋅hc)*dΩ))
        # uc    = CellField(u₀, Ω)
        # e     = u₀-uf
        # err_u = sqrt(sum(∫(e⋅e)*dΩ))/sqrt(sum(∫(uc⋅uc)*dΩ))
        # println("n=", n, ",\terr_u: ", err_u, ",\terr_h: ", err_h)

        #@test abs(err_u - l2_err_u[i]) < 10.0^-12
        #@test abs(err_h - l2_err_h[i]) < 10.0^-12
      end
  end
end
prun_debug(main,mpi,1)

end # module