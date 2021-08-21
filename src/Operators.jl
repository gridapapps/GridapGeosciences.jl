"""
  perp(u,n)

  cross product of manifold's outward unit normal and vector-valued field u
"""
function perp(u,n)
   n×u
end
const ⟂ = perp

"""
Outward unit normal of sphere
"""
function normal_unit_sphere(θϕ)
 θ,ϕ = θϕ
 n=spherical_to_cartesian_matrix(VectorValue(θ,ϕ,1.0))⋅VectorValue(0,0,1)
 n
end

"""
Jacobian of parametric unit sphere
"""
function J_unit_sphere(θϕ)
  θ,ϕ = θϕ
  TensorValue{3,2}(-sin(θ)*cos(ϕ), cos(θ)*cos(ϕ),      0,
                   -sin(ϕ)*cos(θ),-sin(ϕ)*sin(θ), cos(ϕ))
end

"""
First fundamental form parametric unit sphere (a.k.a. metric / Grammian)
"""
function G_unit_sphere(θϕ)
  Jθϕ=J_unit_sphere(θϕ)
  transpose(Jθϕ)⋅Jθϕ
end

"""
Spherical divergence on unit sphere
IMPORTANT NOTE:
* The input function "v" and output function
  take values on the parametric space
"""
function divergence_unit_sphere(v)
  function tmp(θϕ)
     Gθϕ=G_unit_sphere(θϕ)
     Jθϕ=J_unit_sphere(θϕ)
     function f(θϕ)
        sqrt(det(Gθϕ))*inv(Gθϕ)⋅transpose(Jθϕ)⋅(v(θϕ))
     end
     1.0/sqrt(det(Gθϕ))*(∇⋅(f))(θϕ)
  end
end

"""
Spherical Laplacian on unit sphere
IMPORTANT NOTE:
  * The input function "v" and output function
    take values on the parametric space
"""
function laplacian_unit_sphere(v)
  function tmp(θϕ)
     Gθϕ=G_unit_sphere(θϕ)
     Jθϕ=J_unit_sphere(θϕ)
     function f(θϕ)
        sqrt(det(Gθϕ))*inv(Gθϕ)⋅(∇(v)(θϕ))
     end
     1.0/sqrt(det(Gθϕ))*(∇⋅(f))(θϕ)
  end
end
