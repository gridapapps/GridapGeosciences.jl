#module WeakDivPerpTests
   using Gridap
   using GridapGeosciences

   """setup the cubed sphere model in cartesian coordinates"""
   function setup_model(n, mapping)
     # Create domain and test spaces V and Q
     domain = (-1,1,-1,1,-1,1)
     cells  = (n,n,n)
     model  = CartesianDiscreteModel(domain,cells,map=mapping)

     # Restrict model to cube surface (using new BoundaryDiscreteModel)
     labels = get_face_labeling(model)
     bgface_to_mask = Gridap.Geometry.get_face_mask(labels,"boundary",2)
     Γface_to_bgface = findall(bgface_to_mask)
     Dc2Dp3model = Gridap.Geometry.BoundaryDiscreteModel(Polytope{2},model,Γface_to_bgface)
   end


   # Nondivergent velocity field tangent to the sphere
   # Reference:
   #     Lauritzen et. al. GMD 2012
   #     Eq. (18) and (19)
   function W(θϕ)
     θ,ϕ = θϕ
     u = 2*sin(θ)^2*sin(2*ϕ) + 2π/5*cos(ϕ)
     v = 2*sin(2*θ)*cos(ϕ)
     spherical_to_cartesian_matrix(VectorValue(θ,ϕ,1.0))⋅VectorValue(u,v,0)
   end

   """
   Spherical divergence on unit sphere applied to perp(W)
   """
   function divM_perpW(θϕ)
     n_times_W=(θϕ)->(normal_unit_sphere(θϕ)×W(θϕ))
     f=divergence_unit_sphere(n_times_W)
     f(θϕ)
   end

   function compute_error_weak_div_perp(model,order,degree)
     # Setup geometry
     Ω=Triangulation(model)
     dΩ=Measure(Ω,degree)
     n=get_normal_vector(Ω)

     # Setup H(div) spaces
     reffe_rt = ReferenceFE(raviart_thomas, Float64, order)
     V = FESpace(model, reffe_rt ; conformity=:HDiv)
     U = TrialFESpace(V)

     # H1 spaces
     reffe_lgn = ReferenceFE(lagrangian, Float64, order+1)
     S = FESpace(model, reffe_lgn; conformity=:H1)
     R = TrialFESpace(S)

     # Project the analytical velocity field W onto the H(div) space
     a1(u,v) = ∫(v⋅u)dΩ
     b1(v)   = ∫(v⋅(W∘xyz2θϕ))dΩ
     op      = AffineFEOperator(a1,b1,U,V)
     wh      = solve(op)

     e = (W∘xyz2θϕ)-wh
     println(sqrt(sum(∫(e⋅e)dΩ)))

     # Compute weak div_perp operator
     a2(u,v) = ∫(v*u)dΩ

     #b2(v)   = ∫(⟂(∇(v),nsphere∘xyz2θϕ)⋅(wh) )dΩ
     b2(v)   = ∫(⟂(∇(v),normal_unit_sphere∘xyz2θϕ)⋅(wh) )dΩ


     op      = AffineFEOperator(a2,b2,R,S)
     divwh   = solve(op)

     e = divwh-divM_perpW∘xyz2θϕ
     sqrt(sum(∫(e*e)dΩ)),divwh
   end
   model=CubedSphereDiscreteModel(40)
   e,divwh=compute_error_weak_div_perp(model,0,8)
   #model=setup_model(2,GridapGeosciences.map_cube_to_sphere)
   writevtk(Triangulation(model),"divwh",nsubcells=16,cellfields=["divwh"=>divwh,
                                                     "divM_perpW∘xyz2θϕ"=>divM_perpW∘xyz2θϕ,
                                                     "error"=>divwh-divM_perpW∘xyz2θϕ,
                                                     "n"=>get_normal_vector(Triangulation(model))])
#end

# function _grad_perp_phys_domain(model, order, Ω, R, S, U, V, u, qₖ, wₖ)
#   # ∫∇⟂α⋅udΩ
#   # α: Test function,  ∈ H₁(Ω)
#   # u: velocity,       ∈ H(div,Ω)
#   #
#   # arguments:
#   # model: geometry of the domain
#   # order: polynomial degree of the test functions
#   # Ω:     the domain
#   # U:     trial functions
#   # V:     test functions
#   # qₖ:    quadrature points
#   # wₖ:    quadrature weights    # Evaluate the Jacobian at the quadrature points
#   ξₖ  = get_cell_map(model)
#   Jt  = lazy_map(Broadcasting(∇), ξₖ)
#   Jq  = lazy_map(evaluate, Jt, qₖ)

#   # H₁ test functions
#   reffe_lgn    = ReferenceFE(lagrangian, Float64, order+1)
#   basis, reffe_args, reffe_kwargs = reffe_lgn
#   lgn          = ReferenceFE(model, basis, reffe_args...; reffe_kwargs...)

#   # gradient in physical space
#   σ = Gridap.ReferenceFEs.get_shapefuns(lgn[1])
#   ref∇σ = Broadcasting(∇)(σ)
#   ref∇σₖ = Fill(ref∇σ, num_cells(model))
#   phys∇σₖ = lazy_map(Broadcasting(push_∇), ref∇σₖ, ξₖ)
#   phys∇σₖq = lazy_map(evaluate, phys∇σₖ, qₖ)

#   # radial vector in physical space
#   ξₖq=lazy_map(evaluate, ξₖ, qₖ)
#   kq=lazy_map(evaluate,Fill(Broadcasting(radial_vector),num_cells(model)),ξₖq)

#   # grad-perp in physical space
#   kphys∇σₖq = lazy_map(Gridap.Fields.BroadcastingFieldOpMap(×), kq, phys∇σₖq)

#   # evaluate u
#   uq = lazy_map(evaluate, Gridap.CellData.get_data(u), qₖ)

#   intcq = lazy_map(Gridap.Fields.BroadcastingFieldOpMap(⋅), kphys∇σₖq, uq)

#   iwqc  = lazy_map(Gridap.Fields.IntegrationMap(), intcq, wₖ, Jq)
#   assem = SparseMatrixAssembler(U, S)
#   dc    = Gridap.CellData.DomainContribution()
#   Gridap.CellData.add_contribution!(dc, Ω, iwqc)
#   data  = Gridap.FESpaces.collect_cell_vector(S, dc)
#   rhs   = assemble_vector(assem, data)
# end
