function H1MM_downwind_trial_funcs(model, order, qₖ, wₖ, h, u, τ)
  # return the (unassembled) matrix
  # ∫αhαᵈdΩ, ∀α∈ H₁(Ω)
  # where αᵈ is a downwinded trial function, with quadrature points shifted as
  # qₖ → qₖ + τu/|J|
  # reference:
  #     Lee, CAMWA, 2021, 89, 68-77, eq (33)
  ξₖ = get_cell_map(model)
  Jt = lazy_map(Broadcasting(∇), ξₖ)
  Jq = lazy_map(evaluate, Jt, qₖ)

  # H₁ test functions
  reffe_lgn    = ReferenceFE(lagrangian, Float64, order+1)
  basis, reffe_args, reffe_kwargs = reffe_lgn
  lgn          = ReferenceFE(model, basis, reffe_args...; reffe_kwargs...)
  α            = Gridap.ReferenceFEs.get_shapefuns(lgn[1])
  αₖ           = Fill(α, num_cells(model))

  # TODO: we want the velocity vector in LOCAL coordinates! not Piola mapped...
  uq = lazy_map(evaluate, Gridap.CellData.get_data(u), qₖ)
  hq = lazy_map(evaluate, Gridap.CellData.get_data(h), qₖ)

  sqrt_det_JtxJ = lazy_map(Operation(Gridap.TensorValues.meas), Jt)
  qₖ_up  = similar(qₖ)
  qₖ_up .= qₖ .+ τ .* uq ./ sqrt_det_JtxJ

  αq      = lazy_map(evaluate, αₖ, qₖ)
  αq_up   = lazy_map(evaluate, αₖ, qₖ_up)
  αq_up_T = lazy_map(transpose, αq_up)

  αq_up_h = lazy_map(Gridap.Fields.BroadcastingFieldOpMap(*), αq_up, hq)
  intq    = lazy_map(Gridap.Fields.BroadcastingFieldOpMap(*), αq_up_h, αq_up_T)
  iwqc    = lazy_map(Gridap.Fields.IntegrationMap(), intq, wₖ, Jq)
  iwqc, αq_up_T
end

function diagnose_pv_downwind_trial_funcs(model, order, dΩ, qₖ, wₖ, f, h, u, τ, R, S)
  # solve the system:
  #
  # ∫αhqdΩ = -∫∇⟂α⋅udΩ + ∫αfdΩ, ∀α∈ H₁(Ω)
  # where:
  #
  # q : potential vorticity; q = (∇×u + f)/h
  # f : coriolis force (∈ H₁(Ω)
  # h : fluid depth
  # u : velocity
  #
  # order      : polynomial order
  # dΩ         : measure of the elements
  # qₖ         : quadrature points
  # wₖ         : quadrature weights

  # the linear form right hand side
  grad_perp = grad_perp_ref_domain(model, order, u, qₖ, wₖ)
  b(s)  = ∫(s*f)*dΩ
  rhsdc = b(s)
  # subtract the weak form curl as evaluated using the low level API
  Gridap.CellData.add_contribution!(rhsdc, get_triangulation(dΩ.quad), grad_perp)
  # assemble the right hand side
  data  = Gridap.FESpaces.collect_cell_vector(S, rhsdc)
  assem = SparseMatrixAssembler(R, S)
  rhs   = assemble_vector(assem, data)
  # assemble the left hand side
  data, α_up = H1MM_downwind_trial_funcs(model, order, qₖ, wₖ, h, u, τ)
  assem = SparseMatrixAssembler(R, S)
  lhs   = assemble_matrix(assem, data)

  op = AffineFEOperator(R, S, lhs, rhs)
  q  = solve(op)

  q, α_up
end

function RotMat_downwind_trial_funcs(model, order, qₖ, wₖ, q, F, α_up)
  # ∫v×(q*F)dΩ
  # v:  test function,  ∈ H(div,Ω)
  # F:  mass flux,      ∈ H(div,Ω)
  # q:  potential vorticity (at the quadrature points)

  ξₖ  = get_cell_map(model)
  Jt  = lazy_map(Broadcasting(∇), ξₖ)
  Jq  = lazy_map(evaluate, Jt, qₖ)

  qq  = lazy_map(Gridap.Fields.BroadcastingFieldOpMap(*), α_up, q)

  # RT test functions
  vₖ  = Geofem.RT_pullback(model, order)
  vₖq = lazy_map(evaluate, vₖ, qₖ)
  Fq  = lazy_map(evaluate, Gridap.CellData.get_data(F), qₖ)
  Fqq = lazy_map(Gridap.Fields.BroadcastingFieldOpMap(*), qq, Fq)

  ξₖq     = lazy_map(evaluate, ξₖ, qₖ)
  # radial vector in physical space at quadrature points
  kq      = lazy_map(evaluate, Fill(Broadcasting(Geofem.radial_vector), num_cells(model)), ξₖq)
  # k×Fq (in physical space)
  k_x_Fq  = lazy_map(Gridap.Fields.BroadcastingFieldOpMap(×), kq, Fqq)
  intvxFq = lazy_map(Gridap.Fields.BroadcastingFieldOpMap(⋅), vₖq, k_x_Fq)
  iwqc    = lazy_map(Gridap.Fields.IntegrationMap(), intvxFq, wₖ, Jq)
end

function grad_perp_ref_domain(model, order, u, qₖ, wₖ)
  # ∫∇⟂α⋅udΩ
  # α: Test function,  ∈ H₁(Ω)
  # u: velocity,       ∈ H(div,Ω)
  #
  # arguments:
  # model: geometry of the domain
  # order: polynomial degree of the test functions
  # qₖ:    quadrature points
  # wₖ:    quadrature weights    # Evaluate the Jacobian at the quadrature points
  ξₖ  = get_cell_map(model)
  Jt  = lazy_map(Broadcasting(∇), ξₖ)
  Jq  = lazy_map(evaluate, Jt, qₖ)

  # H₁ test functions
  reffe_lgn    = ReferenceFE(lagrangian, Float64, order+1)
  basis, reffe_args, reffe_kwargs = reffe_lgn
  lgn          = ReferenceFE(model, basis, reffe_args...; reffe_kwargs...)
  α            = Gridap.ReferenceFEs.get_shapefuns(lgn[1])
  αₖ           = Fill(α, num_cells(model))
  # map the H₁ test functions into H(div) via the ∇⟂ operator in the reference element
  grad_perp_αₖ = lazy_map(Broadcasting(grad_perp), αₖ)

  # get the index of the panel for each element
  fl       = get_face_labeling(model)
  panel_id = fl.d_to_dface_to_entity[3]
  # grad perp turns out to have reverse orientation in half the panels
  panel_flip = ones(Bool, (length(grad_perp_αₖ), length(grad_perp_αₖ[1])))
  for i in 1:length(grad_perp_αₖ)
    panel_flip[i,:] .= true
    if panel_id[i] == 25 || panel_id[i] == 21 || panel_id[i] == 24
      panel_flip[i,:] .= false
    end
  end
  fpanel_flip   = lazy_map(Broadcasting(Gridap.Fields.ConstantField), panel_flip)
  # pull back the H(div) test functions into global coordinates
  m             = Gridap.ReferenceFEs.ContraVariantPiolaMap()
  sqrt_det_JtxJ = lazy_map(Operation(Gridap.TensorValues.meas), Jt)
  ϕₖs   = lazy_map(Broadcasting(Operation(m)), grad_perp_αₖ, Jt, sqrt_det_JtxJ, fpanel_flip)
  uq    = lazy_map(evaluate, Gridap.CellData.get_data(u), qₖ)
  ϕₖsq  = lazy_map(evaluate, ϕₖs, qₖ)
  intcq = lazy_map(Gridap.Fields.BroadcastingFieldOpMap(⋅), ϕₖsq, uq)
  iwqc  = lazy_map(Gridap.Fields.IntegrationMap(), intcq, wₖ, Jq)
  -1.0*iwqc
end

function diagnose_vorticity!(model, order, Ω, qₖ, wₖ, H1MMchol, u, w)
  # ∇×u, weak form: ∫ααdΩ^{-1}∫-∇⟂α⋅udΩ; ∀α∈ H₁(Ω)
  iwqc  = grad_perp_ref_domain(model, order, u, qₖ, wₖ)
  assem = SparseMatrixAssembler(U, S)
  dc    = Gridap.CellData.DomainContribution()
  Gridap.CellData.add_contribution!(dc, Ω, iwqc)
  data  = Gridap.FESpaces.collect_cell_vector(S, dc)
  rhs   = assemble_vector(assem, data)
  copy!(get_free_dof_values(w), rhs)
  ldiv!(H1MMchol, get_free_dof_values(w))
end

function diagnose_potential_vorticity(model, order, dΩ, qₖ, wₖ, f, h, u, R, S)
  # solve the system:
  #
  # ∫αhqdΩ = -∫∇⟂α⋅udΩ + ∫αfdΩ, ∀α∈ H₁(Ω)
  # where:
  #
  # q : potential vorticity; q = (∇×u + f)/h
  # f : coriolis force (∈ H₁(Ω)
  # h : fluid depth
  # u : velocity
  #
  # order      : polynomial order
  # dΩ         : measure of the elements
  # qₖ         : quadrature points
  # wₖ         : quadrature weights

  # the bilinear form left hand side
  r      = get_trial_fe_basis(R)
  s      = get_fe_basis(S)
  a(r,s) = ∫(s*h*r)*dΩ
  # the linear form right hand side
  grad_perp = grad_perp_ref_domain(model, order, u, qₖ, wₖ)
  b(s)  = ∫(s*f)*dΩ
  rhsdc = b(s)
  # subtract the weak form curl as evaluated using the low level API
  Gridap.CellData.add_contribution!(rhsdc, get_triangulation(dΩ.quad), grad_perp)
  # assemble the right hand side
  data  = Gridap.FESpaces.collect_cell_vector(S, rhsdc)
  assem = SparseMatrixAssembler(R, S)
  rhs   = assemble_vector(assem, data)
  # assemble the left hand side
  data  = Gridap.FESpaces.collect_cell_matrix(R, S, a(r,s))
  assem = SparseMatrixAssembler(R, S)
  lhs   = assemble_matrix(assem, data)

  op = AffineFEOperator(R, S, lhs, rhs)
  q  = solve(op)
end
