function _fm(f, m)
   function fm(m)
     αβ -> begin
         x = m(αβ)
         f(x)
     end
   end
end

 deriv_sqrt= x -> 0.5/sqrt(x)
 function deriv_det(x::SymTensorValue{2})
   Gridap.TensorValues.SymTensorValue(x[2,2],-x[2,1],x[1,1])
 end
 function deriv_det(x::SymTensorValue{3})
   Gridap.TensorValues.SymTensorValue(
     x[2,2]*x[3,3] - x[2,3]^2,
     x[2,3]*x[1,3] - x[1,2]*x[3,3],
     x[1,2]*x[2,3] - x[2,2]*x[1,3],
     x[1,1]*x[3,3] - x[1,3]^2,
     x[1,2]*x[1,3] - x[1,1]*x[2,3],
     x[1,1]*x[2,2] - x[1,2]^2,
   )
 end

cpAB = (A,B)->contracted_product(Val(2), A, permutedims(B,(2,3,1)))
function _Δs_no_ad(f, Ω_atlas)
  # surflap(f::Function) = m -> surflap(f,m)
  # surflap(f::Function,m::Field) = αβ -> 1/sqrtg(m,αβ) * ( divergence(W(f,m))(αβ) )
  # W(f::Function,m::Field) = αβ ->  sqrtg(m,αβ)*( inv_metric(m,αβ) ⋅ gradient(f(m))(αβ) )

  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  f_cf = f∘ambient_map_cf
  metric_cf = MetricCellField(Ω_atlas)
  inv_metric_cf = InvMetricCellField(Ω_atlas)
  meas_cf = MeasureCellField(Ω_atlas)
  covariant_basis_cf = transpose∘∇(ambient_map_cf)
  gradient_f_cf = (∇(f)∘ambient_map_cf)⋅covariant_basis_cf

  ## BEGIN Machinery to compute gradient(meas_cf)
  # v_l = A_ij * B_kij
  grad_meas_cf = (deriv_sqrt∘det∘metric_cf)*
                   Operation(cpAB)(deriv_det∘metric_cf,gradient(metric_cf))
  ## END Machinery to compute gradient(meas_cf)

  ## BEGIN Machinery to compute gradient_gradient(f_cf)
  # A_ij = v_k * B_ijk
  cpvB=(v,B)->contracted_product(Val(1), v, permutedims(B,(3,1,2)))
  gradient_gradient_cf = ∇(ambient_map_cf)⋅(∇∇(f)∘ambient_map_cf)⋅covariant_basis_cf +
                                            Operation(cpvB)(∇(f)∘ambient_map_cf,
                                            ∇∇(ambient_map_cf))
  ## END Machinery to compute gradient_gradient(f_cf)

  # w_cf = meas_cf*(inv_metric_cf⋅gradient_f_cf)
  # divergence(w_cf) =
  #   grad(meas_cf)⋅( inv_metric_cf⋅gradient_f_cf ) + (1)
  #   meas_cf*divergence(inv_metric_cf⋅gradient_f_cf) (2+3) =
  #   grad(meas_cf)⋅( inv_metric_cf⋅gradient_f_cf ) +   (1)
  #   meas_cf*divergence(inv_metric_cf)⋅gradient_f_cf + (2)
  #   meas_cf*(inv_metric_cf ⊙ gradient(gradient_f_cf)) (3)
  div_wcf_first_term = grad_meas_cf⋅(inv_metric_cf⋅gradient_f_cf)
  div_wcf_second_term = meas_cf*(divergence(inv_metric_cf)⋅gradient_f_cf)
  div_wcf_third_term = meas_cf*(inv_metric_cf ⊙ gradient_gradient_cf)
  div_wcf = div_wcf_first_term + div_wcf_second_term + div_wcf_third_term
  1.0/meas_cf * div_wcf
end

function _Δs_ad(f, Ω_atlas)
  # surflap(f::Function) = m -> surflap(f,m)
  # surflap(f::Function,m::Field) = αβ -> 1/sqrtg(m,αβ) * ( divergence(W(f,m))(αβ) )
  # W(f::Function,m::Field) = αβ ->  sqrtg(m,αβ)*( inv_metric(m,αβ) ⋅ gradient(f(m))(αβ) )
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  ambient_maps = Gridap.CellData.get_data(ambient_map_cf)
  cell_field = lazy_map(m->GenericField(surflap(_fm(f,m),m)),ambient_maps)
  CellData.GenericCellField(cell_field,Ω_atlas,PhysicalDomain())
end

## f is an scalar-valued ambient-space function
function Δs(f::Function,
            Ω_atlas::BFTATDMIM{Dc,Dc,Da,G,A,P,C,O};
            use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
    use_automatic_differentiation ? _Δs_ad(f, Ω_atlas) : _Δs_no_ad(f, Ω_atlas)
end

function Δs(f::Function,
            Ω_atlas::AdaptedTriangulation{Dc,Da,<:BFTATDMIM{Dc,Dc,Da,G,A,P,C,O}};
            use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
    Δs_trian = use_automatic_differentiation ? _Δs_ad(f, Ω_atlas.trian) : _Δs_no_ad(f, Ω_atlas.trian)
    Gridap.CellData.GenericCellField(get_data(Δs_trian), Ω_atlas, Gridap.CellData.DomainStyle(Δs_trian))
end

function _compose(parametric_space_quantity, inv_ambient_map_cell_field)
    # Not able to do Δs_parametric_space ∘ InvAmbientMapCellField(Ω_atlas) with Gridap
    # I perform the composition manually with lazy_map below as a workaround.
    parametric_space_data = Gridap.CellData.get_data(parametric_space_quantity)
    inv_ambient_map_data = Gridap.CellData.get_data(inv_ambient_map_cell_field)
    composed_data = lazy_map(∘, parametric_space_data, inv_ambient_map_data)
    CellData.GenericCellField(composed_data,
                              Gridap.Geometry.get_triangulation(inv_ambient_map_cell_field),
                              Gridap.CellData.PhysicalDomain())
end

function Δs(f::Function,
            Ω_atlas::BFTATDMEM{Dc,Dc,Da,G,A,P,C,O};
            use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
    Δs_parametric_space = use_automatic_differentiation ? _Δs_ad(f, Ω_atlas) : _Δs_no_ad(f, Ω_atlas)
    _compose(Δs_parametric_space, InvAmbientMapCellField(Ω_atlas))
end

function Δs(f::Function,
            Ω_atlas::AdaptedTriangulation{Dc,Da,<:BFTATDMEM{Dc,Dc,Da,G,A,P,C,O}};
            use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
    Δs_parametric_space = use_automatic_differentiation ? _Δs_ad(f, Ω_atlas) : _Δs_no_ad(f, Ω_atlas)
    _compose(Δs_parametric_space, InvAmbientMapCellField(Ω_atlas))
end

# returns the contravariant component of surface gradient
function _∇s_no_ad(f, Ω_atlas)
  # sgrad(f::Function) = m -> sgrad(f,m)
  # sgrad(f::Function,m::Field) = αβ -> (inv_metric(m,αβ) ⋅ gradient(f(m))(αβ) )
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  inv_metric_cf = InvMetricCellField(Ω_atlas)
  covariant_basis_cf = transpose∘∇(ambient_map_cf)
  gradient_f_cf = (∇(f)∘ambient_map_cf)⋅covariant_basis_cf
  (inv_metric_cf⋅gradient_f_cf)
end

# returns the contravariant component of surface gradient
function _∇s_ad(f, Ω_atlas)
  # sgrad(f::Function) = m -> sgrad(f,m)
  # sgrad(f::Function,m::Field) = αβ -> (inv_metric(m,αβ) ⋅ gradient(f(m))(αβ) )
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  ambient_maps = Gridap.CellData.get_data(ambient_map_cf)
  cell_field = lazy_map(m->GenericField(sgrad(_fm(f,m),m)),ambient_maps)
  CellData.GenericCellField(cell_field,Ω_atlas,PhysicalDomain())
end

# returns the contravariant component of surface gradient
function ∇s(f::Function,
            Ω_atlas::BFTATDMIM{Dc,Dc,Da,G,A,P,C,O};
            use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  use_automatic_differentiation ? _∇s_ad(f, Ω_atlas) : _∇s_no_ad(f, Ω_atlas)
end

# returns the contravariant component of surface gradient
function ∇s(f::Function,
            Ω_atlas::AdaptedTriangulation{Dc,Da,<:BFTATDMIM{Dc,Dc,Da,G,A,P,C,O}};
            use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  ∇s_trian = use_automatic_differentiation ? _∇s_ad(f, Ω_atlas.trian) : _∇s_no_ad(f, Ω_atlas.trian)
  Gridap.CellData.GenericCellField(get_data(∇s_trian), Ω_atlas, Gridap.CellData.DomainStyle(∇s_trian))
end

function ∇s(f::Function,
            Ω_atlas::BFTATDMEM{Dc,Dc,Da,G,A,P,C,O};
            use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  ∇s_parametric_space_contra = use_automatic_differentiation ? _∇s_ad(f, Ω_atlas) : _∇s_no_ad(f, Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  covariant_basis_cf = transpose∘∇(ambient_map_cf)
  ∇s_parametric_space = covariant_basis_cf ⋅ ∇s_parametric_space_contra
  _compose(∇s_parametric_space, InvAmbientMapCellField(Ω_atlas))
end

function ∇s(f::Function,
            Ω_atlas::AdaptedTriangulation{Dc,Da,<:BFTATDMEM{Dc,Dc,Da,G,A,P,C,O}};
            use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  ∇s_parametric_space_contra = use_automatic_differentiation ? _∇s_ad(f, Ω_atlas) : _∇s_no_ad(f, Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  covariant_basis_cf = transpose∘∇(ambient_map_cf)
  ∇s_parametric_space = covariant_basis_cf ⋅ ∇s_parametric_space_contra
  _compose(∇s_parametric_space, InvAmbientMapCellField(Ω_atlas))
end

# return the contravariant componet
function _skew_∇s_no_ad(f, Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  meas_cf = MeasureCellField(Ω_atlas)
  J_cf = transpose∘∇(ambient_map_cf)
  grad_f_cf = (∇(f)∘ambient_map_cf)⋅J_cf
  skew_grad_parametric = (perp∘grad_f_cf)*(1.0/meas_cf)
end

# return the contravariant componet
function _skew_∇s_ad(f, Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  ambient_maps = Gridap.CellData.get_data(ambient_map_cf)
  cell_field = lazy_map(m->GenericField(skew_surfgrad(_fm(f,m),m)),ambient_maps)
  CellData.GenericCellField(cell_field,Ω_atlas,PhysicalDomain())
end

# return the contravariant componet
function skew_∇s(f::Function, Ω_atlas::BFTATDMIM{2,2,Da,G,A,P,C,O};
                   use_automatic_differentiation=false) where {Da, G, A, P, C, O}
   use_automatic_differentiation ? _skew_∇s_ad(f, Ω_atlas) : _skew_∇s_no_ad(f, Ω_atlas)
end

# return the contravariant componet
function skew_∇s(f::Function,
                  Ω_atlas::AdaptedTriangulation{2,2,<:BFTATDMIM{2,2,Da,G,A,P,C,O}};
                  use_automatic_differentiation=false) where {Da, G, A, P, C, O}
  skew_∇s_trian = use_automatic_differentiation ? _skew_∇s_ad(f, Ω_atlas.trian) : _skew_∇s_no_ad(f, Ω_atlas.trian)
  Gridap.CellData.GenericCellField(get_data(skew_∇s_trian), Ω_atlas, Gridap.CellData.DomainStyle(skew_∇s_trian))
end

function skew_∇s(f::Function, Ω_atlas::BFTATDMIM{Dc,Dc,Da,G,A,P,C,O};
                   use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  @notimplemented "skew_∇s is only implemented for 2D surfaces"
end

function skew_∇s(f::Function, Ω_atlas::BFTATDMEM{Dc,Dc,Da,G,A,P,C,O};
                   use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  @notimplemented "skew_∇s is only implemented for intrinsic manifolds"
end


# Contravariant components of 3D vector vecX
# The contravariatn mapping is  ̃u = J u
# so u = J^† ̃u
contra_v(vecX::Function,m::Field) = αβ -> pinvJ(J(m,αβ))⋅vecX(m)(αβ)
contra_v(vecX::Function) = p -> contra_v(vecX,p)

function _divs_ad(f, Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  ambient_maps = Gridap.CellData.get_data(ambient_map_cf)
  cell_field = lazy_map(m->GenericField(surfdiv(contra_v(_fm(f,m)),m)),ambient_maps)
  CellData.GenericCellField(cell_field,Ω_atlas,PhysicalDomain())
end

function _divs_no_ad(f, Ω_atlas)
    # 1/m * div( m * (J^†⋅(f∘ϕ)) ), where J^†=inv(g)⋅Jᵀ
    # grad(m)⋅(J^†⋅(f∘ϕ)) +
    # div((J^†⋅(f∘ϕ))) = tr(grad(J^†):(f∘ϕ)) + tr(J^†⋅grad(f∘ϕ))
    # grad(J^†) = grad(inv(g)⋅Jᵀ) = grad(inv(g))⋅Jᵀ + inv(g)⊙grad(Jᵀ)
    metric_cf = MetricCellField(Ω_atlas)
    meas_cf = MeasureCellField(Ω_atlas)
    inv_metric_cf = InvMetricCellField(Ω_atlas)
    ambient_map_cf = AmbientMapCellField(Ω_atlas)
    grad_ambient_map_cf = ∇(ambient_map_cf)
    f_cf = f∘ambient_map_cf
    grad_f_cf = ∇(f)∘ambient_map_cf
    Jt_cf = ∇(ambient_map_cf)

    # grad(inv(g))⋅Jᵀ
    grad_inv_metric_cf = ∇(inv_metric_cf)
    trace_1=Operation(tr)((grad_inv_metric_cf⋅Jt_cf)⋅f_cf)

    # inv(g)⋅grad(Jᵀ)
    trace_2 = Operation(tr)((inv_metric_cf ⋅ ∇(Jt_cf))⋅f_cf)

    # tr(J^†⋅grad(f∘ϕ)) = tr((inv(g)⋅Jᵀ)⋅grad(f∘ϕ))
    trace_3=Operation(tr)((inv_metric_cf⋅grad_ambient_map_cf)⋅
                              ((grad_f_cf)⋅(transpose∘grad_ambient_map_cf)))

    grad_meas_cf = (deriv_sqrt∘det∘metric_cf)*
                   Operation(cpAB)(deriv_det∘metric_cf,gradient(metric_cf))


    return (1.0/meas_cf)*(meas_cf*(trace_1+trace_2+trace_3) +
                           grad_meas_cf⋅(inv_metric_cf⋅Jt_cf⋅f_cf))
end

function _skew_divs_no_ad(f, Ω_atlas)
    # -1/m * div( m^2 * inv(g) R(J^†⋅(f∘ϕ)) ), where J^†=inv(g)⋅Jᵀ
    # div( m^2 * inv(g) R(J^†⋅(f∘ϕ)) )
    # div( m^2 * inv(g) R(J^†⋅(f∘ϕ)) ) =
    #   grad(m^2)⋅(inv(g) R(J^†⋅(f∘ϕ))) + m^2 * div(inv(g) R(J^†⋅(f∘ϕ)))
    # div(inv(g) R(J^†⋅(f∘ϕ))) = tr(grad(inv(g))⋅R(J^†⋅(f∘ϕ))) + tr(inv(g)⋅grad(R(J^†⋅(f∘ϕ))))
    #
    Gridap.Helpers.@notimplemented "skew_divs without automatic differentiation is not implemented yet"
end

function _skew_divs_ad(f, Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  ambient_maps = Gridap.CellData.get_data(ambient_map_cf)
  cell_field = lazy_map(m->GenericField(skew_surfdiv(contra_v(_fm(f,m)),m)),ambient_maps)
  CellData.GenericCellField(cell_field,Ω_atlas,PhysicalDomain())
end

# Surface divergence of an ambient vector-valued function which is
# pulled back using the pseudo-inverse of the jacobian of the ambient
# map without multiplying by the measure
function divs(f::Function, Ω_atlas::BFTATDMIM{Dc,Dc,Da,G,A,P,C,O};
              use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
   use_automatic_differentiation ? _divs_ad(f, Ω_atlas) : _divs_no_ad(f, Ω_atlas)
end

function divs(f::Function,
              Ω_atlas::AdaptedTriangulation{Dc,Da,<:BFTATDMIM{Dc,Dc,Da,G,A,P,C,O}};
              use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  divs_trian = use_automatic_differentiation ? _divs_ad(f, Ω_atlas.trian) : _divs_no_ad(f, Ω_atlas.trian)
  Gridap.CellData.GenericCellField(get_data(divs_trian), Ω_atlas, Gridap.CellData.DomainStyle(divs_trian))
end

function divs(f::Function,
              Ω_atlas::BFTATDMEM{Dc,Dc,Da,G,A,P,C,O};
              use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  ∇s_parametric_space = use_automatic_differentiation ? _divs_ad(f, Ω_atlas) : _divs_no_ad(f, Ω_atlas)
  _compose(∇s_parametric_space, InvAmbientMapCellField(Ω_atlas))
end

function divs(f::Function,
              Ω_atlas::AdaptedTriangulation{Dc,Da,<:BFTATDMEM{Dc,Dc,Da,G,A,P,C,O}};
              use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  ∇s_parametric_space = use_automatic_differentiation ? _divs_ad(f, Ω_atlas.trian) : _divs_no_ad(f, Ω_atlas.trian)
  _compose(∇s_parametric_space, InvAmbientMapCellField(Ω_atlas))
end


function skew_divs(f::Function, Ω_atlas::BFTATDMIM{2,2,Da,G,A,P,C,O};
                   use_automatic_differentiation=true) where {Da, G, A, P, C, O}
   use_automatic_differentiation ? _skew_divs_ad(f, Ω_atlas) : _skew_divs_no_ad(f, Ω_atlas)
end

function skew_divs(f::Function,
                   Ω_atlas::AdaptedTriangulation{Dc,Da,<:BFTATDMIM{Dc,Dc,Da,G,A,P,C,O}};
                   use_automatic_differentiation=true) where {Dc, Da, G, A, P, C, O}
  skew_divs_trian = use_automatic_differentiation ? _skew_divs_ad(f, Ω_atlas.trian) : _skew_divs_no_ad(f, Ω_atlas.trian)
  Gridap.CellData.GenericCellField(get_data(skew_divs_trian), Ω_atlas, Gridap.CellData.DomainStyle(skew_divs_trian))
end

function skew_divs(f::Function,
                   Ω_atlas::BFTATDMEM{2,2,Da,G,A,P,C,O};
                   use_automatic_differentiation=true) where {Da, G, A, P, C, O}
  ∇s_parametric_space = use_automatic_differentiation ? _skew_divs_ad(f, Ω_atlas) : _skew_divs_no_ad(f, Ω_atlas)
  _compose(∇s_parametric_space, InvAmbientMapCellField(Ω_atlas))
end

function skew_divs(f::Function,
                   Ω_atlas::AdaptedTriangulation{Dc,Da,<:BFTATDMEM{2,2,Da,G,A,P,C,O}};
                   use_automatic_differentiation=true) where {Dc, Da, G, A, P, C, O}
  ∇s_parametric_space = use_automatic_differentiation ? _skew_divs_ad(f, Ω_atlas.trian) : _skew_divs_no_ad(f, Ω_atlas.trian)
  _compose(∇s_parametric_space, InvAmbientMapCellField(Ω_atlas))
end

function skew_divs(f::Function, Ω_atlas::BFTATDMIM{Dc,Dc,Da,G,A,P,C,O};
                   use_automatic_differentiation=true) where {Dc, Da, G, A, P, C, O}
  @notimplemented "skew_divs is only implemented for 2D surfaces"
end

function skew_divs(f::Function, Ω_atlas::BFTATDMEM{Dc,Dc,Da,G,A,P,C,O};
                   use_automatic_differentiation=true) where {Dc, Da, G, A, P, C, O}
  @notimplemented "skew_divs is only implemented for 2D surfaces"
end

dagger(vec::Function) = m -> dagger(vec,m)
dagger(vec::Function,m::Field) = αβ ->  J(m)(αβ)⋅(inv_metric(m,αβ)⋅perp( contra_v(vec(m))(αβ) )) * sqrtg(m,αβ)

function _dagger_ad(f::Function, Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  ambient_maps = Gridap.CellData.get_data(ambient_map_cf)
  cell_field = lazy_map(m->GenericField(dagger(_fm(f,m),m)),ambient_maps)
  CellData.GenericCellField(cell_field,Ω_atlas,PhysicalDomain())
end

function _dagger_no_ad(f::Function, Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  f_cf = f∘ambient_map_cf
  measure_cf = MeasureCellField(Ω_atlas)
  inv_metric_cf = InvMetricCellField(Ω_atlas)
  J_cf = transpose∘∇(ambient_map_cf)
  f_cf_parametric = (pinvJ∘J_cf)⋅f_cf
  J_cf⋅(inv_metric_cf⋅(perp∘f_cf_parametric))*measure_cf
end

function dagger(f::Function, Ω_atlas::BFTATDMIM{2,2,Da,G,A,P,C,O};
                use_automatic_differentiation=false) where {Da, G, A, P, C, O}
  use_automatic_differentiation ? _dagger_ad(f, Ω_atlas) : _dagger_no_ad(f, Ω_atlas)
end

function dagger(f::Function, Ω_atlas::BFTATDMIM{Dc,Dc,Da,G,A,P,C,O};
                use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  @notimplemented "dagger is only implemented for intrinsic 2D surfaces"
end

function dagger(f::Function, Ω_atlas::BFTATDMEM{Dc,Dc,Da,G,A,P,C,O};
                use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  @notimplemented "dagger is only implemented for intrinsic 2D surfaces"
end

Jt(m) = x -> transpose(J(m,x))
Jtu(u,m) = x -> Jt(m)(x)⋅u(m)(x)
# Returns the co-vector associated to the surface curl of a vector-valued field
curls(u,m) = x-> 1.0/sqrtg(m,x)*metric(m,x)⋅curl(Jtu(u,m))(x)
# Returns the co-vector associated to the surface curl of the surface curl of a vector-valued field
curls_curls(u, m) = x -> 1.0/sqrtg(m,x)*metric(m,x)⋅curl(curls(u,m))(x)

## surface divergence
_divs(u,m) = x -> sqrtg(m)(x)*inv(J(m,x))⋅u(m)(x)
divs(u, m) = x -> 1/sqrtg(m)(x)*(divergence(_divs(u,m))(x))

### covariant vector surfgrad(surfdiv u)
grads_divs(u, m) = x-> gradient(divs(u,m))(x)
vec_laps(u,m) = x -> grads_divs(u,m)(x) - curls_curls(u,m)(x)

function _vecΔs_ad(f, Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  ambient_maps = Gridap.CellData.get_data(ambient_map_cf)
  cell_field = lazy_map(m->GenericField(vec_laps(_fm(f,m),m)),ambient_maps)
  CellData.GenericCellField(cell_field,Ω_atlas,PhysicalDomain())
end

function _vecΔs_no_ad(f, Ω_atlas)
  Gridap.Helpers.@notimplemented "vecΔs without automatic differentiation is not implemented yet"
end

# Returns the co-vector components of the vector surface laplacian applied to the
# ambient vector-valued function f
function vecΔs(f::Function, Ω_atlas::BFTATDMIM{Dc,Dc,Da,G,A,P,C,O};
                use_automatic_differentiation=true) where {Dc, Da, G, A, P, C, O}
  use_automatic_differentiation ? _vecΔs_ad(f, Ω_atlas) : _vecΔs_no_ad(f, Ω_atlas)
end

function vecΔs(f::Function, Ω_atlas::AdaptedTriangulation{Dc,Da,<:BFTATDMIM{Dc,Dc,Da,G,A,P,C,O}};
                use_automatic_differentiation=true) where {Dc, Da, G, A, P, C, O}
  vecΔs_trian = use_automatic_differentiation ? _vecΔs_ad(f, Ω_atlas.trian) : _vecΔs_no_ad(f, Ω_atlas.trian)
  Gridap.CellData.GenericCellField(get_data(vecΔs_trian), Ω_atlas, Gridap.CellData.DomainStyle(vecΔs_trian))
end

# ### Curl of covariant components of u
# ucov(u,m,x) = Jt(m)(x)⋅u(m)(x)
# ucov(u,m) = x -> ucov(u,m,x)
# curl_ucov(u,m,x) = curl(ucov(u,m))(x)
# curl_ucov(u,m) = x -> curl_ucov(u,m,x)

# ### Covariant components of surfcurl u
# _curls(u,m,x) = 1.0/sqrtg(m,x)*metric(m,x)⋅curl_ucov(u,m,x)
# _curls(u,m) = x -> _curls(u,m,x)
# curls(u,m,x) = curl(_curls(u,m))(x)
# curls(u,m) = x -> curls(u,m,x)
function _curls_ad(f, Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  ambient_maps = Gridap.CellData.get_data(ambient_map_cf)
  cell_field = lazy_map(m->GenericField(curls(_fm(f,m),m)),ambient_maps)
  CellData.GenericCellField(cell_field,Ω_atlas,PhysicalDomain())
end

function _curls_no_ad(f, Ω_atlas)
  Gridap.Helpers.@notimplemented "curls without automatic differentiation is not implemented yet"
end

# Returns the co-vector components of the surface curl operator applied to the
# ambient vector-valued function f
function curls(f::Function, Ω_atlas::BFTATDMIM{Dc,Dc,Da,G,A,P,C,O};
                use_automatic_differentiation=true) where {Dc, Da, G, A, P, C, O}
  use_automatic_differentiation ? _curls_ad(f, Ω_atlas) : _curls_no_ad(f, Ω_atlas)
end

function curls(f::Function, Ω_atlas::AdaptedTriangulation{Dc,Da,<:BFTATDMIM{Dc,Dc,Da,G,A,P,C,O}};
                use_automatic_differentiation=true) where {Dc, Da, G, A, P, C, O}
  curls_trian = use_automatic_differentiation ? _curls_ad(f, Ω_atlas.trian) : _curls_no_ad(f, Ω_atlas.trian)
  Gridap.CellData.GenericCellField(get_data(curls_trian), Ω_atlas, Gridap.CellData.DomainStyle(curls_trian))
end





################################################################################
## The below operators are required for surface Stokes. Some are repeated above
## Need to decide how to gather these properly
################################################################################
# Contravariant components of the ∇^perp (∇^perp ⋅ ̃u)
# = J*    1/√g R gradient ( -1/√g div(R g u)     )
# where u is contravariant component of ̃u, Recall Rg = (√g)^2 g^{-1} R
_my_skew_sdiv(f, m) = αβ -> detg(m,αβ)*inv_metric(m,αβ)⋅(perp(contra_v(f,m)(αβ)))
my_skew_surfdiv(f, m) = αβ ->  -1.0/sqrtg(m,αβ)*(divergence(_my_skew_sdiv(f,m))(αβ))
curly_curl(u, m) = x -> 1/sqrtg(m)(x)*perp(gradient(my_skew_surfdiv(u,m))(x))


# Contravariant component of ∇(∇⋅ ̃u)
# = J *    g^{-1} gradient ( 1/√g div( √g u)   )
# where u is contravariant component of ̃u
_my_divs(u,m) = x -> sqrtg(m)(x)*contra_v(u,m)(x)
my_divs(u, m) = x -> 1/sqrtg(m)(x)*(divergence(_my_divs(u,m))(x))

my_grads_divs(u, m) = x-> gradient(my_divs(u,m))(x)
contra_grads_divs(u, m) = x-> inv_metric(m,x)⋅my_grads_divs(u,m)(x)

# Contravariant component of the surface vector laplacian in 2D
vec_laps_2D(u,m) = x ->  contra_grads_divs(u,m)(x) -1.0*curly_curl(u,m)(x)

function _vecΔs_ad_2D(f, Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  ambient_maps = Gridap.CellData.get_data(ambient_map_cf)
  cell_field = lazy_map(m->GenericField(vec_laps_2D(_fm(f,m),m)),ambient_maps)
  CellData.GenericCellField(cell_field,Ω_atlas,PhysicalDomain())
end

function _vecΔs_no_ad_2D(f, Ω_atlas)
  Gridap.Helpers.@notimplemented "vecΔs_2D without automatic differentiation is not implemented yet"
end

# Returns the contravariant components of the vector surface laplacian applied to the
# ambient vector-valued function f in 2D
function vecΔs_2D(f::Function, Ω_atlas::BFTATDMIM{Dc,Dc,Da,G,A,P,C,O};
                use_automatic_differentiation=true) where {Dc, Da, G, A, P, C, O}
  use_automatic_differentiation ? _vecΔs_ad_2D(f, Ω_atlas) : _vecΔs_no_ad_2D(f, Ω_atlas)
end


function vecΔs_2D(f::Function,
              Ω_atlas::AdaptedTriangulation{Dc,Da,<:BFTATDMIM{Dc,Dc,Da,G,A,P,C,O}};
              use_automatic_differentiation=true) where {Dc, Da, G, A, P, C, O}
  vecΔs_2D_trian = use_automatic_differentiation ? _vecΔs_ad_2D(f, Ω_atlas) : _vecΔs_no_ad_2D(f, Ω_atlas)
  Gridap.CellData.GenericCellField(get_data(vecΔs_2D_trian), Ω_atlas, Gridap.CellData.DomainStyle(vecΔs_2D_trian))
end


# Return the contravariant components of the surface gradient
# = J *    g^{-1} gradient(f)
sgrad_contra(f::Function,m::Field) = αβ ->  (inv_metric(m,αβ) ⋅ gradient(f(m))(αβ) )
sgrad_contra(f::Function) = m -> sgrad_contra(f,m)

function _contra_∇s_no_ad(f, Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  inv_metric_cf = InvMetricCellField(Ω_atlas)
  covariant_basis_cf = transpose∘∇(ambient_map_cf)
  gradient_f_cf = (∇(f)∘ambient_map_cf)⋅covariant_basis_cf
  inv_metric_cf⋅gradient_f_cf
end

function _contra_∇s_ad(f, Ω_atlas)
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  ambient_maps = Gridap.CellData.get_data(ambient_map_cf)
  cell_field = lazy_map(m->GenericField(sgrad_contra(_fm(f,m),m)),ambient_maps)
  CellData.GenericCellField(cell_field,Ω_atlas,PhysicalDomain())
end


function ∇s_contra(f::Function,
            Ω_atlas::BFTATDMIM{Dc,Dc,Da,G,A,P,C,O};
            use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  use_automatic_differentiation ? _contra_∇s_ad(f, Ω_atlas) : _contra_∇s_no_ad(f, Ω_atlas)
end

function ∇s_contra(f::Function,
            Ω_atlas::AdaptedTriangulation{Dc,Da,<:BFTATDMIM{Dc,Dc,Da,G,A,P,C,O}};
            use_automatic_differentiation=false) where {Dc, Da, G, A, P, C, O}
  ∇s_trian = use_automatic_differentiation ? _contra_∇s_ad(f, Ω_atlas.trian) : _contra_∇s_no_ad(f, Ω_atlas.trian)
  Gridap.CellData.GenericCellField(get_data(∇s_trian), Ω_atlas, Gridap.CellData.DomainStyle(∇s_trian))
end
