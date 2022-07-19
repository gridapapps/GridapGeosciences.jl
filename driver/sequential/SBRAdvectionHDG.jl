module SBRAdvectionHDG

using Gridap
using GridapGeosciences
using GridapHybrid

function Gridap.Geometry.push_normal(invJt,n)
  v = invJt⋅n
  m = sqrt(inner(v,v))
  if m < eps()
    return zero(v)
  else
    return v/m
  end
end

# solid body rotation on the sphere for the scalar flux form
# advection equation using a hybridized discontinuous Galerkin
# method for L2 elements

order  = 1
degree = 3
n      = 4
dt     = 0.5*2.0*π/((order+1)*4*n)
# single rotation about the sphere
nstep  = Int(2.0*π/dt)

# solid body rotation velocity field
function u₀(xyz)
  θϕr = xyz2θϕr(xyz)
  u   = cos(θϕr[2])
  spherical_to_cartesian_matrix(θϕr)⋅VectorValue(u,0,0)
end

# Gaussian tracer initial condition
function p₀(xyz)
  rsq = (xyz[1] - 1.0)*(xyz[1] - 1.0) + xyz[2]*xyz[2] + xyz[3]*xyz[3]
  exp(-4.0*rsq)
end

model = CubedSphereDiscreteModel(n,2; radius=1)

advection_hdg(model, order, degree,
              u₀, p₀, dt, nstep;
              write_solution=true,
              write_solution_freq=Int(nstep/8),
              write_diagnostics=true,
              write_diagnostics_freq=1,
              dump_diagnostics_on_screen=true)
end