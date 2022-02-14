"""copy a FEFunction on a FE Space"""
clone_fe_function(space,f)=FEFunction(space,copy(get_free_dof_values(f)))

"""
 Given an arbitrary number of FEFunction arguments,
 returns a tuple with their corresponding free DoF Values
"""
function Gridap.get_free_dof_values(functions...)
  map(get_free_dof_values,functions)
end

function setup_mixed_spaces(model, order)
  reffe_rt  = ReferenceFE(raviart_thomas, Float64, order)
  V = FESpace(model, reffe_rt ; conformity=:HDiv)
  U = TrialFESpace(V)
  reffe_lgn = ReferenceFE(lagrangian, Float64, order)
  Q = FESpace(model, reffe_lgn; conformity=:L2)
  P = TrialFESpace(Q)
  reffe_lgn = ReferenceFE(lagrangian, Float64, order+1)
  S = FESpace(model, reffe_lgn; conformity=:H1)
  R = TrialFESpace(S)

  R, S, U, V, P, Q
end

function setup_and_factorize_mass_matrices(dΩ,
                                           R, S, U, V, P, Q;
                                           mass_matrix_solver=BackslashSolver())
  amm(a,b) = ∫(a⋅b)dΩ
  H1MM   = assemble_matrix(amm, R, S)
  RTMM   = assemble_matrix(amm, U, V)
  L2MM   = assemble_matrix(amm, P, Q)

  ssH1MM = symbolic_setup(mass_matrix_solver,H1MM)
  nsH1MM = numerical_setup(ssH1MM,H1MM)

  ssRTMM = symbolic_setup(mass_matrix_solver,RTMM)
  nsRTMM = numerical_setup(ssRTMM,RTMM)

  ssL2MM = symbolic_setup(mass_matrix_solver,L2MM)
  nsL2MM = numerical_setup(ssL2MM,L2MM)

  H1MM, RTMM, L2MM, nsH1MM, nsRTMM, nsL2MM
end

function new_vtk_step(Ω,file,_cellfields)
  n = size(_cellfields)[1]
  createvtk(Ω,
            file,
            cellfields=_cellfields,
            nsubcells=n)
end
