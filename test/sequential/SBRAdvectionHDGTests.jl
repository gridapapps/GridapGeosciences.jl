module SBRAdvectionHDGTests

using Test
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

# solid body rotation on the sphere for the scalar flux form
# advection equation using a hybridized discontinuous Galerkin
# method for L2 elements

order  = 1
degree = 3
l2_err_i   = [2.990077828e-01 , 9.625369737e-02 , 1.918189011e-02 ]
enst_con_i = [-2.533366481e-01, -7.330337526e-02, -1.289758067e-02]

for i in 1:3
  n      = 2*2^i
  dt     = 0.5*2.0*π/((order+1)*4*n)
  # single rotation about the sphere
  nstep  = Int(2.0*π/dt)

  model = CubedSphereDiscreteModel(n,2; radius=1)

  l2_err, mass_con, enst_con = advection_hdg(model, order, degree,
                                             u₀, p₀, dt, nstep;
                                             write_solution=false,
                                             write_solution_freq=Int(nstep/8),
                                             write_diagnostics=true,
                                             write_diagnostics_freq=1,
                                             dump_diagnostics_on_screen=true)
  @test abs(l2_err - l2_err_i[i])     < 10.0^-10
  @test abs(mass_con)                 < 10.0^-12
  @test abs(enst_con - enst_con_i[i]) < 10.0^-10
end
end
