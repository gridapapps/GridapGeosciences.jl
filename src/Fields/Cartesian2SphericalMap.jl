struct Cartesian2SphericalMap <: Field
end

Gridap.Fields.Broadcasting(f::Cartesian2SphericalMap) = f

"""
The forward map goes 3D -> 2D.
  θ = atan(Y,X)
  ϕ = atan(Z,sqrt(X^2 + Y^2))

  Note, this map must be applied cellwise to obtain correct latlon.
  This is because if any(X > 0) and any(Y < 0) within the cell, take the θ angle
  from the negative side.
  Applying the map this way means that we account for the position of the cell within
  the whole sphere to obtain the correct longitude.
  Thus, the single point version of the map is not available
    ** update, have to make it available for vtk (unsure why)
"""

function Gridap.Arrays.return_cache(f::Cartesian2SphericalMap,cellx::AbstractArray{<:VectorValue{3}})
  out = similar(cellx,VectorValue{2,Float64})
  return out
end


function Gridap.Arrays.evaluate!(cache,f::Cartesian2SphericalMap,cellx::AbstractArray{<:VectorValue{3}} )
  out = cache

  x = map(x->x[1],cellx)
  y = map(x->x[2],cellx)
  z = map(x->x[3],cellx)

  ## This hack is because sometimes at higher refinements, the y is slightly negative/positive
  ## when it really should be zero. It is numerical error. So to overcome, set
  ## the abs(y) <1e-16 (machine eps) to be zero
  idx = abs.(y) .<1e-16
  if any(idx)
    y[idx].= 0.0
  end

  r = sqrt.(x.^2 + y.^2 + z.^2)
  θ = rem2pi.(atan.(y, x),RoundDown)
  ϕ = asin.(z./r)

  # if there are negative ys and positive xs
  if any(y .< 0 ) && any(x .> 0)
    θ = 2*π .+  rem2pi.(atan.(y, x),RoundUp)
  end



  # map!((x,y)->VectorValue(x,y)  ,out, θ,ϕ)
  out = map(θ,ϕ) do θ,ϕ
    VectorValue(θ,ϕ)
  end

  return out
end

function Gridap.Arrays.return_type(f::Cartesian2SphericalMap,x::VectorValue{3})
  out = VectorValue{2,Float64}
  return out
end

function Gridap.Arrays.evaluate!(cache,f::Cartesian2SphericalMap,x::VectorValue{3})
  out = cache
  @check false """\n The Cartesian2SphericalMap cannot be evaluated on a single point.
  This is because we need to know the position of the cell in ambient 3D space to properly
  handle the periodicity of angles. Conseqeuently, evalaute Cartesian2SphericalMap on
  an AbstractArray of cellwise points. Alternatively, implement a variant of Cartesian2SphericalMap
  that uses the panel index to properly handle the periodicity of angles.
  """
end
