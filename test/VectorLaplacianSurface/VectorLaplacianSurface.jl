"""
Vector Laplacian on the 2D cubed sphere, intrinsic (chart-space) formulation.

Solves, for a tangent field u on the sphere Σ,

    α u - ν( gradg(divg u) + gradg⊥(divgperp u) ) = f     on Σ (closed manifold, no boundary)

where gradg/divg are the ordinary intrinsic surface gradient/divergence and
gradg⊥/divgperp are their skew ("rotated") counterparts (Paper2_GridapGeoscience,
article-theory, §2.2 eq. (skew ops), §4.4 eq. (surface stokes) viscous+reaction terms):

    gradg  f = g⁻¹ ∇f
    divg   v = (1/√g) ∇·(√g v)
    gradg⊥ f = (1/√g) R∇f ,  R = [[0,-1],[1,0]]
    divgperp v = (1/√g)(∂v₂/∂x¹ - ∂v₁/∂x²)     (v = covariant components g·(contravariant v))

Discretized with the grad-conforming vector-valued H1 Lagrangian space (contravariant
nodal DOFs glued across the 6 cube-panel charts via GridapGeosciences' transmission-map
machinery), i.e. exactly the paper's 𝕎(𝒯) velocity space.

Manufactured solution: u = gradg⊥(Ψ) with Ψ(x,y,z) = x*y, a degree-2 spherical harmonic
(harmonic in R^3, hence a Laplace-Beltrami eigenfunction on the sphere with eigenvalue
-l(l+1)/R^2 = -6/R^2). Using the identities divg(gradg⊥ Ψ) ≡ 0 and
divgperp(gradg⊥ Ψ) = Δ_g Ψ (the stream-function/vorticity relation) collapses the forcing
to a closed form with no extra differentiation: f = (α + 6ν/R²) u.
"""

module VectorLaplacianSurfaceTests

using Gridap
using Gridap.Helpers
using Gridap.Algebra
using GridapGeosciences
using Test

# Ψ(x,y,z) = x*y : harmonic in R^3 (degree-2 spherical harmonic, l=2)
ΨX(xyz) = xyz[1]*xyz[2]

function solve_vector_laplacian(atlas_model,
      p_fe::Int, dir::String, ν::Float64, α::Float64, radius::Float64,
      ls=LUSolver(), return_vtk=false; _i_am_main=true)

  lvl = nref(atlas_model)
  _i_am_main && println("VectorLaplacianSurface: p_fe = $(p_fe); nref = $lvl; ν = $ν; α = $α")

  degree = 4*(p_fe+1)

  Ω_atlas = Triangulation(atlas_model)
  dΩ = Measure(Ω_atlas,degree)
  dΩ_error = Measure(Ω_atlas,2*degree)

  metric_cf = MetricCellField(Ω_atlas)
  meas_cf   = MeasureCellField(Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  covariant_basis_cf = transpose∘∇(ambient_map_cf)

  # ∇(√g) and ∇g via Jacobi's formula / analytic metric gradient (same chain rule
  # GridapGeosciences uses internally in SurfaceDiffOps.jl's _Δs_no_ad/_divs_no_ad;
  # reproduced here since those are non-exported package internals). Both divg and
  # divgperp are built by hand-applying the product rule to already-differentiated
  # pieces (native ∇v, native gradient(metric_cf)) rather than asking Gridap to
  # differentiate a product CellField directly — Gridap has no registered product
  # rule for a SymTensorValue⋅VectorValue dot product.
  _deriv_sqrt(x) = 0.5/sqrt(x)
  _deriv_det(x::Gridap.TensorValues.SymTensorValue{2}) = Gridap.TensorValues.SymTensorValue(x[2,2],-x[2,1],x[1,1])
  _cpAB(A,B) = Gridap.TensorValues.contracted_product(Val(2), A, permutedims(B,(2,3,1)))
  grad_metric_cf = gradient(metric_cf)  # ThirdOrderTensorValue, [i,j,k] = ∂g_jk/∂x_i
  grad_meas_cf = (_deriv_sqrt∘det∘metric_cf)*Operation(_cpAB)(_deriv_det∘metric_cf,grad_metric_cf)

  # [i,j] = Σ_k v^k ∂g_jk/∂x_i   (product-rule term from differentiating g⋅v)
  _cpvB(v,B) = Gridap.TensorValues.contracted_product(Val(1), v, permutedims(B,(3,1,2)))
  _grad2curl(T) = T[1,2]-T[2,1]

  divg(v)     = divergence(v) + (v⋅grad_meas_cf)*(1.0/meas_cf)
  function divgperp(v)
    dgv = Operation(_cpvB)(v,grad_metric_cf) + ∇(v)⋅metric_cf   # = ∇(g⋅v)
    Operation(_grad2curl)(dgv)*(1.0/meas_cf)
  end

  # exact solution: u = gradg⊥(Ψ), contravariant components
  u_exact_ambient_cf = skew_∇s(ΨX,Ω_atlas;use_automatic_differentiation=false)
  u_exact_contra_cf  = pinvJ∘covariant_basis_cf ⋅ u_exact_ambient_cf

  # forcing: f = (α + 6ν/R²) u   (Δ_g Ψ = -6/R² Ψ for this degree-2 harmonic)
  λ = α + 6.0*ν/radius^2
  f_contra_cf = λ*u_exact_contra_cf

  V = TestFESpace(Ω_atlas, ReferenceFE(lagrangian,VectorValue{2,Float64},p_fe); conformity=:H1)
  U = TrialFESpace(V)

  a(u,ψ) = ∫( ν*(divg(u)*divg(ψ) + divgperp(u)*divgperp(ψ))*meas_cf
              + α*(u⋅(metric_cf⋅ψ))*meas_cf )dΩ
  l(ψ)   = ∫( (f_contra_cf⋅(metric_cf⋅ψ))*meas_cf )dΩ

  op = AffineFEOperator(a,l,U,V)
  uh = solve(ls,op)

  _e = u_exact_contra_cf - uh
  el2_u = sqrt(sum(∫( (_e⋅(metric_cf⋅_e))*meas_cf )dΩ_error))

  _i_am_main && println("eu = $(el2_u)")

  if return_vtk
    cellfields = ["u"=>covariant_basis_cf⋅u_exact_contra_cf,
                  "uh"=>covariant_basis_cf⋅uh,
                  "eu"=>covariant_basis_cf⋅(u_exact_contra_cf-uh)]
    writevtk_with_cell_geomap(ambient_map_cf,Ω_atlas,dir*"/ambient_model_nref$(lvl)_p$(p_fe)",
          cellfields=cellfields,append=false)
  end

  return el2_u, false, false
end

################################################################################
#### Auto convergence test
################################################################################
function main(models::AbstractArray; ps=[1,2], ν=1.0, α=0.0, radius=1.0, ls=LUSolver(), _i_am_main=true)
  dir = @__DIR__
  p_convergence_auto_test(ps,models,solve_vector_laplacian,dir,ν,α,radius,ls;_i_am_main=_i_am_main)
end

end # module
