
using Gridap
using GridapGeosciences
using Test


forward_maps = map(p->CubedSphereForwardMap(p),collect(1:6))
jacobians_transpose = map(m->gradient(m),forward_maps)
jacobians = map(m->Operation(transpose)(gradient(m)),forward_maps)


_panel_to_cartesian(fX::Function,p::Int) = γαβ -> fX(my_map(p)(γαβ))
_panel_to_cartesian(fX::Function) = p -> _panel_to_cartesian(fX,p)

forward_pinv_jacobian_3D(p) = αβ -> GridapGeosciences.Helpers.pinvJ(my_J(p)(αβ))
J_3D(p::Int,γαβ) = my_J(p)(γαβ)

Jt_3D(p::Int,γαβ) =  transpose(J_3D(p,γαβ))
metric_3D(p::Int,γαβ) = Jt_3D(p,γαβ)⋅J_3D(p,γαβ)
inv_metric_3D(p::Int,γαβ) = inv(metric_3D(p,γαβ))
detg_3D(p::Int,γαβ) = det(metric_3D(p,γαβ))
sqrtg_3D(p::Int,γαβ) = sqrt(detg_3D(p,γαβ))

metric_3D(p::Int) = γαβ -> metric_3D(p,γαβ)
inv_metric_3D(p::Int) = γαβ ->  inv_metric_3D(p,γαβ)
detg_3D(p::Int)  = γαβ -> detg_3D(p,γαβ)
sqrtg_3D(p::Int) = γαβ -> sqrtg_3D(p,γαβ)
grad_meas_3D(p::Int) = γαβ -> gradient(sqrtg_3D(p))(γαβ)

W_3D(f::Function,p::Int) = γαβ ->  sqrtg_3D(p,γαβ)*( inv_metric_3D(p,γαβ) ⋅ gradient(f(p))(γαβ))
surflap_3D(f::Function,p::Int) = γαβ -> 1/sqrtg_3D(p,γαβ) * ( divergence(W_3D(f,p))(γαβ) )
surflap_3D(f::Function) = p -> surflap_3D(f,p)

contr_gradf_3D(f::Function,p::Int) = γαβ -> inv_metric_3D(p,γαβ) ⋅ gradient(f(p))(γαβ)
contr_gradf_3D(f::Function) = p -> contr_gradf_3D(f,p)

sgrad_3D(f::Function,p::Int) =  γαβ -> J_3D(p,γαβ) ⋅ (inv_metric_3D(p,γαβ) ⋅ gradient(f(p))(γαβ))
sgrad_3D(f::Function) = p -> sgrad_3D(f,p)

_a_sdiv_3D(vec::Function,p) = γαβ ->  sqrtg_3D(p,γαβ)*( vec(p)(γαβ))
surfdiv_3D(vec::Function,p::Int) = γαβ -> 1/sqrtg_3D(p,γαβ) * ( divergence(_a_sdiv_3D(vec,p))(γαβ) )
surfdiv_3D(vec::Function) = p -> surfdiv_3D(vec,p)


############## Testing
fX(XYZ::VectorValue{3}) = XYZ[1]*XYZ[2]*XYZ[3]
αβ = Point(π/4,π/4)
γαβ = Point(1.0,π/4,π/4)
p = 1

mapps = [forward_map_2D, forward_map_3D]
jacs = [ forward_jacobian_2D, forward_jacobian]
pts = [αβ, γαβ]

 for (_my_map,_my_J,pt) in zip(mapps,jacs,pts)
  f = panel_to_cartesian(fX)
  _f =  _panel_to_cartesian(fX)

  global my_map = _my_map
  global my_J = _my_J
  @test f(p)(pt) == _f(p)(pt)
  @test forward_pinv_jacobian(p)(pt) == forward_pinv_jacobian_3D(p)(pt)
  @test metric(p,pt) == metric_3D(p,pt)
  @test inv_metric(p,pt) == inv_metric_3D(p,pt)
  @test sqrtg(p,pt) == sqrtg_3D(p,pt)
  @test sgrad(f)(p)(pt) == sgrad_3D(f)(p)(pt) == sgrad(_f)(p)(pt)
  @test surflap(f)(p)(pt) == surflap_3D(f)(p)(pt) == surflap(_f)(p)(pt)

  return true
end

# ############## vector projection
# # contravariat components of 3D vector vecX
# auto_contra_v(vecX::Function,p::Int) = x -> auto_forward_pinv_jacobian(p)(x)⋅ vecX(p)(x)
# auto_contra_v(vecX::Function) = p -> auto_contra_v(vecX,p)



# # projection of 3D vector vecX
# auto_projection_v(vecX::Function,p::Int) = x -> auto_forward_jacobian(p)(x) ⋅ contra_v(vecX,p)(x)
# auto_projection_v(vecX::Function) = p -> auto_projection_v(vecX,p)


# include("../Advection/advection_funcs.jl")

# vX2 = panel_to_cartesian(tangent_vec(vecX))
# _vX2 = _panel_to_cartesian(tangent_vec(vecX))

# vX3 = panel3_to_cartesian(tangent_vec(vecX))
# _vX3 = _panel_to_cartesian(tangent_vec(vecX))

# contra_v(vX2)(p)(αβ) == auto_contra_v(vX2)(p)(αβ) == auto_contra_v(_vX2)(p)(αβ)
# auto_contra_v(vX3)(p)(γαβ) ==  auto_contra_v(_vX3)(p)(γαβ)
