"""
 Convert cartesian coordinates to spherical coordinates (θ,ϕ,r)
 θ, ϕ are in radians
 ϕ is the latitude, i.e., the angle from the equator.
"""
function xyz2θϕr(x)
  r = sqrt(x[1]^2 + x[2]^2 + x[3]^2)
  θ = atan(x[2], x[1])
  ϕ = asin(x[3]/r)
  VectorValue(θ,ϕ,r)
end
function xyz2θϕ(x)
  θ = atan(x[2], x[1])
  ϕ = asin(x[3])
  VectorValue(θ,ϕ)
end


"""
Matrix transformation from spherical vector field to Cartesian vector field.
  θ∈(0,2π)
  ϕ∈(-π/2,π/2)
  r=constant for the sphere
"""
function spherical_to_cartesian_matrix(θϕr)
  θ,ϕ,r = θϕr
  TensorValue(-sin(θ)       , cos(θ)       ,      0,
              -sin(ϕ)*cos(θ),-sin(ϕ)*sin(θ), cos(ϕ),
               cos(ϕ)*cos(θ), cos(ϕ)*sin(θ), sin(ϕ))
end

"""
Matrix transformation from cartesian vector field to spherical vector field
"""
function cartesian_to_spherical_matrix(xyz)
  x,y,z = xyz
  sr = sqrt(x^2+y^2+z^2)
  cr = sqrt(x^2+y^2)
  TensorValue(-y/cr, (x*z)/(sr*cr), x/sr,
               x/cr, (y*z)/(sr*cr), y/sr,
                  0,   -cr/(sr*cr), z/sr)
end
