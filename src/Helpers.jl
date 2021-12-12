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

function setup_and_factorize_mass_matrices(dΩ, R, S, U, V, P, Q)
  amm(a,b) = ∫(a⋅b)dΩ
  H1MM = assemble_matrix(amm, R, S)
  RTMM = assemble_matrix(amm, U, V)
  L2MM = assemble_matrix(amm, P, Q)
  H1MMchol = lu(H1MM)
  RTMMchol = lu(RTMM)
  L2MMchol = lu(L2MM)

  H1MM, RTMM, L2MM, H1MMchol, RTMMchol, L2MMchol
end

function new_vtk_step(Ω,file,_cellfields)
  n = size(_cellfields)[1]
  createvtk(Ω,
            file,
            cellfields=_cellfields,
            nsubcells=n)
end

import Gridap.FESpaces: TrialBasis, SingleFieldFEBasis, SingleFieldFEFunction, ContraVariantPiolaMap
import Gridap.Fields: linear_combination, ConstantField, GenericField
import Gridap.CellData: GenericCellField
import Gridap.Arrays: LazyArray

function _undo_piola_map(f::LazyArray{<:Fill{Broadcasting{Operation{ContraVariantPiolaMap}}}})
  ϕrgₖ       = f.args[1]
  fsign_flip = f.args[4]
  fsign_flip=lazy_map(Broadcasting(Operation(x->(-1)^x)), fsign_flip)
  lazy_map(Broadcasting(Operation(*)),fsign_flip,ϕrgₖ)
end

"""
 qh :: Trial functions potential vorticity space
 u  :: RT space FE Function
"""
function upwind_trial_functions(
  qh::SingleFieldFEBasis{<:TrialBasis},
  uh::SingleFieldFEFunction,
  τ::Real,model)
  uh_data=Gridap.CellData.get_data(uh)
  qh_data=Gridap.CellData.get_data(qh)
  rt_trial_ref=_undo_piola_map(uh_data.args[2])
  uh_data_ref=lazy_map(linear_combination,uh_data.args[1],rt_trial_ref)

  ξₖ = get_cell_map(model)
  Jt = lazy_map(Broadcasting(∇), ξₖ)
  sqrt_det_JtxJ = lazy_map(Operation(Gridap.TensorValues.meas), Jt)

  cfτ=Fill(ConstantField(τ),length(uh_data))                                 # τ
  m=Broadcasting(Operation(*))
  cfτ_mul_uh_data_ref=lazy_map(m,cfτ,uh_data_ref)                            # τ*u

  d=Broadcasting(Operation(/))
  cfτ_mul_uh_data_ref_div_meas=lazy_map(d,cfτ_mul_uh_data_ref,sqrt_det_JtxJ) # τ*u/sqrt(JᵀJ)

  xi = Fill(GenericField(identity),length(uh_data))                          # ξ
  m=Broadcasting(Operation(-))
  xi_minus_cfτ_mul_uh_data_ref=lazy_map(m,xi,cfτ_mul_uh_data_ref_div_meas)   # ξ - τ*u/sqrt(JᵀJ)
  cell_field=lazy_map(Broadcasting(∘),qh_data,xi_minus_cfτ_mul_uh_data_ref)  # qh ∘ (ξ - τ*u/sqrt(JᵀJ))
  GenericCellField(cell_field,get_triangulation(qh),ReferenceDomain())
end

function upwind_test_functions(
  qh::SingleFieldFEBasis,
  uh::SingleFieldFEFunction,
  τ::Real,model)
  uh_data=Gridap.CellData.get_data(uh)
  qh_data=Gridap.CellData.get_data(qh)
  rt_trial_ref=_undo_piola_map(uh_data.args[2])
  uh_data_ref=lazy_map(linear_combination,uh_data.args[1],rt_trial_ref)

  ξₖ = get_cell_map(model)
  Jt = lazy_map(Broadcasting(∇), ξₖ)
  sqrt_det_JtxJ = lazy_map(Operation(Gridap.TensorValues.meas), Jt)

  cfτ=Fill(ConstantField(τ),length(uh_data))                                 # τ
  m=Broadcasting(Operation(*))
  cfτ_mul_uh_data_ref=lazy_map(m,cfτ,uh_data_ref)                            # τ*u

  d=Broadcasting(Operation(/))
  cfτ_mul_uh_data_ref_div_meas=lazy_map(d,cfτ_mul_uh_data_ref,sqrt_det_JtxJ) # τ*u/sqrt(JᵀJ)

  xi = Fill(GenericField(identity),length(uh_data))                          # ξ
  m=Broadcasting(Operation(-))
  xi_minus_cfτ_mul_uh_data_ref=lazy_map(m,xi,cfτ_mul_uh_data_ref_div_meas)   # ξ - τ*u/sqrt(JᵀJ)
  cell_field=lazy_map(Broadcasting(∘),qh_data,xi_minus_cfτ_mul_uh_data_ref)  # qh ∘ (ξ - τ*u/sqrt(JᵀJ))
  GenericCellField(cell_field,get_triangulation(qh),ReferenceDomain())
end
