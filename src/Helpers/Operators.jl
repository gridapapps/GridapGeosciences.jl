
J(m::Field,x) = transpose(∇(m)(x))
Jt(m::Field,x) = transpose(J(m,x))
metric(m::Field,x) = Jt(m,x)⋅J(m,x)
inv_metric(m::Field,x) = inv(metric(m,x))
detg(m::Field,x)  = det(metric(m,x))
sqrtg(m::Field,x)  = sqrt(detg(m,x))
forward_jacobian(m::Field,x) = J(m,x)
forward_pinv_jacobian(m::Field,x) = pinvJ(J(m,x))


J(m::Field) = x -> J(m,x)
Jt(m::Field) = x -> Jt(m,x)
metric(m::Field) = x -> metric(m,x)
inv_metric(m::Field) = x -> inv_metric(m,x)
detg(m::Field)  = x -> detg(m,x)
sqrtg(m::Field)  = x -> sqrtg(m,x)
covariant_basis(m::Field) = x -> J(m,x)
forward_jacobian(m::Field) = x -> J(m,x)
forward_pinv_jacobian(m::Field) = x -> pinvJ(J(m,x))

function pinvJ(J::MultiValue{Tuple{D1,D2}}) where {D1,D2}
  @check D2 < D1 ## J = 3x2
  Jt = transpose(J)
  inv(Jt⋅J)⋅Jt
end

function pinvJ(J::MultiValue{Tuple{D,D}}) where D
  inv(J)
end

####### surface laplacin
surflap(f::Function) = m -> surflap(f,m)
surflap(f::Function,m::Field) = αβ -> 1/sqrtg(m,αβ) * ( divergence(W(f,m))(αβ) )
W(f::Function,m::Field) = αβ ->  sqrtg(m,αβ)*( inv_metric(m,αβ) ⋅ gradient(f(m))(αβ) )

####### sgrad
sgrad(f::Function) = m -> sgrad(f,m)
sgrad(f::Function,m::Field) = αβ -> J(m,αβ) ⋅ (inv_metric(m,αβ) ⋅ gradient(f(m))(αβ) )

####### surf div
_sdiv(f::Function,m::Field) = αβ ->  sqrtg(m,αβ)*( f(m)(αβ))
surfdiv(vec::Function) = m -> surfdiv(vec,m)
surfdiv(f::Function,m::Field) = αβ -> 1/sqrtg(m,αβ) * ( divergence(_sdiv(f,m))(αβ) )

####### skew surf div

# grad(m^2)⋅(inv(g) R(J^†⋅(f∘ϕ)))

_skew_sdiv(f::Function,m::Field) = αβ -> detg(m,αβ)*inv_metric(m,αβ)⋅(perp(f(m)(αβ)))
skew_surfdiv(vec::Function) = m -> skew_surfdiv(vec,m)
skew_surfdiv(f::Function,m::Field) = αβ ->  -1.0/sqrtg(m,αβ)*(divergence(_skew_sdiv(f,m))(αβ))

####### skew surf grad

skew_surfgrad(vec::Function) = m -> skew_surfgrad(vec,m)
skew_surfgrad(f::Function,m::Field) = αβ ->  1.0/sqrtg(m,αβ)*J(m)(αβ)⋅perp(gradient(f(m))(αβ))



"""
perp

computes u^⟂ = R u , where u is only defined for 2D parametric models.
This function will fail if the background model is a 3D parametric model,
or 2/3D ambient model
"""

function perp(vec::VectorValue{2})
  VectorValue(-vec[2],vec[1])
end  

function perp(t::TensorValue{2,2})
  TensorValue(-t[2,1],t[1,1],-t[2,2],t[1,2])
end