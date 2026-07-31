using GridapGeosciences
using Gridap

using Gridap.CellData
using Gridap.Fields
using Gridap.Adaptivity

import GridapGeosciences.CellData: grads_divs, _fm, contra_v
import GridapGeosciences.Helpers: J,inv_metric, metric, sqrtg, perp, detg
import GridapGeosciences.Geometry: BFTATDMIM, BFTATDM

perp_metric(m::Field) = x -> perp(metric(m,x))

function PerpMetric(Ω_atlas::BFTATDMIM{Dc,Dc,Da,G,A,P,C,O}) where {Dc, Da, G, A, P, C, O}
  ambient_map_cf = AmbientMapCellField(Ω_atlas)
  ambient_maps = Gridap.CellData.get_data(ambient_map_cf)
  cell_field = lazy_map(m->GenericField(perp_metric(m)),ambient_maps)
  CellData.GenericCellField(cell_field,Ω_atlas,PhysicalDomain())
end

function PerpMetric(
    trian :: AdaptedTriangulation{Dc,Dp,<:BFTATDM{Dc,Dp}},
) where {Dc,Dp}
  cf = PerpMetric(trian.trian)
  Gridap.CellData.GenericCellField(get_data(cf), trian, Gridap.CellData.DomainStyle(cf))
end

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
