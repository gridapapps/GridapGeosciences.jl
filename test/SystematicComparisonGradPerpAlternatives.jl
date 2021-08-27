module systematic_comparison_grad_perp_alternatives
  using Gridap
  using GridapGeosciences
  using FillArrays

  include("ConvergenceAnalysisTools.jl")

  """Compute the grad perp of an H₁ reference element.
  Note that this directly maps the H₁ test functions into H(div)"""
  function grad_perp(α)
    grad_α = ∇(α)
    A = TensorValue{2,2}(0, -1, 1, 0)
    A⋅grad_α
  end

  function radial_vector(xyz)
    x,y,z=xyz
    θϕr = xyz2θϕr(xyz)
    A=spherical_to_cartesian_matrix(θϕr)
    A⋅VectorValue(0,0,1)
  end


  function grad_perp_ref_domain(model, order, Ω, R, S, U, V, u, qₖ, wₖ)
     # ∫∇⟂α⋅udΩ
     # α: Test function,  ∈ H₁(Ω)
     # u: velocity,       ∈ H(div,Ω)
     #
     # arguments:
     # model: geometry of the domain
     # order: polynomial degree of the test functions
     # Ω:     the domain
     # U:     trial functions
     # V:     test functions
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
       panel_flip[i,:] .= false
       if panel_id[i] == 25 || panel_id[i] == 21 || panel_id[i] == 24
         panel_flip[i,:] .= true
       end
     end
     fpanel_flip   = lazy_map(Broadcasting(Gridap.Fields.ConstantField), panel_flip)
     # pull back the H(div) test functions into global coordinates
     m             = Gridap.ReferenceFEs.ContraVariantPiolaMap()
     sqrt_det_JtxJ = lazy_map(Operation(Gridap.TensorValues.meas), Jt)
     ϕₖs   = lazy_map(Broadcasting(Operation(m)),
                                   grad_perp_αₖ,
                                   Jt,
                                   sqrt_det_JtxJ,
                                   fpanel_flip)
     uq    = lazy_map(evaluate, Gridap.CellData.get_data(u), qₖ)
     ϕₖsq  = lazy_map(evaluate, ϕₖs, qₖ)
     intcq = lazy_map(Gridap.Fields.BroadcastingFieldOpMap(⋅), ϕₖsq, uq)
     iwqc  = lazy_map(Gridap.Fields.IntegrationMap(), intcq, wₖ, Jq)
  end

  function grad_perp_phys_domain(model, order, Ω, R, S, U, V, u, qₖ, wₖ)
    # ∫∇⟂α⋅udΩ
    # α: Test function,  ∈ H₁(Ω)
    # u: velocity,       ∈ H(div,Ω)
    #
    # arguments:
    # model: geometry of the domain
    # order: polynomial degree of the test functions
    # Ω:     the domain
    # U:     trial functions
    # V:     test functions
    # qₖ:    quadrature points
    # wₖ:    quadrature weights    # Evaluate the Jacobian at the quadrature points
    ξₖ  = get_cell_map(model)
    Jt  = lazy_map(Broadcasting(∇), ξₖ)
    Jq  = lazy_map(evaluate, Jt, qₖ)

    # H₁ test functions
    reffe_lgn    = ReferenceFE(lagrangian, Float64, order+1)
    basis, reffe_args, reffe_kwargs = reffe_lgn
    lgn          = ReferenceFE(model, basis, reffe_args...; reffe_kwargs...)

    # gradient in physical space
    σ = Gridap.ReferenceFEs.get_shapefuns(lgn[1])
    ref∇σ = Broadcasting(∇)(σ)
    ref∇σₖ = Fill(ref∇σ, num_cells(model))
    phys∇σₖ = lazy_map(Broadcasting(Gridap.Fields.push_∇), ref∇σₖ, ξₖ)
    phys∇σₖq = lazy_map(evaluate, phys∇σₖ, qₖ)

    # radial vector in physical space
    ξₖq=lazy_map(evaluate, ξₖ, qₖ)
    kq=lazy_map(evaluate,Fill(Broadcasting(radial_vector),num_cells(model)),ξₖq)

    # grad-perp in physical space
    kphys∇σₖq = lazy_map(Gridap.Fields.BroadcastingFieldOpMap(×), kq, phys∇σₖq)

    # evaluate u
    uq = lazy_map(evaluate, Gridap.CellData.get_data(u), qₖ)

    intcq = lazy_map(Gridap.Fields.BroadcastingFieldOpMap(⋅), kphys∇σₖq, uq)

    iwqc  = lazy_map(Gridap.Fields.IntegrationMap(), intcq, wₖ, Jq)
  end


  function compare_grad_perp_alternatives(model,order,degree)
    RT = ReferenceFE(raviart_thomas, Float64, order  )
    CG = ReferenceFE(lagrangian    , Float64, order+1)
    V  = FESpace(model,RT; conformity=:Hdiv)
    E  = FESpace(model,CG; conformity=:H1)
    U  = TrialFESpace(V)
    F  = TrialFESpace(E)

    Ω = Triangulation(model)
    dΩ = Measure(Ω,degree)

    u=FEFunction(V,ones(num_free_dofs(V)))
    n=get_normal_vector(model)
    a(v)=∫(perp(∇(v),n)⋅u)dΩ
    b1=assemble_vector(a,E)

    quad_cell_point = get_cell_points(dΩ.quad)
    qₖ = Gridap.CellData.get_data(quad_cell_point)
    wₖ = dΩ.quad.cell_weight
    #iwqc=grad_perp_phys_domain(model, order, Ω, E,F,U,V,u, qₖ, wₖ)
    iwqc=grad_perp_ref_domain(model, order, Ω, E,F,U,V,u, qₖ, wₖ)
    dc    = Gridap.CellData.DomainContribution()
    Gridap.CellData.add_contribution!(dc, Ω, iwqc)
    data  = Gridap.FESpaces.collect_cell_vector(E, dc)
    assem = SparseMatrixAssembler(U,E)
    b2 = assemble_vector(assem, data)
    norm(b2-b1)/norm(b2)
  end

  # Testing cubed sphere mesh with analytical geometric mapping
  @time hs0,k0errors,_=convergence_study(compare_grad_perp_alternatives,
                                         generate_n_values(2;n_max=30),2,4)
  k0errors

end
