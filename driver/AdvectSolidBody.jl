module AdvectSolidBody

using Gridap
using GridapGeosciences

function u₀(xyz)
  θϕr = xyz2θϕr(xyz)
  x,y,z = xyz
  spherical_to_cartesian_matrix(θϕr)⋅VectorValue(z/rₑ,0,0)
end

function q₀(xyz)
  θϕr   = xyz2θϕr(xyz)
  θ,ϕ,r = θϕr
  α     = 2.0/3.0
  β     = α

  cos(ϕ)*exp(-1.0*(θ/α)*(θ/α))*exp(-1.0*(ϕ/β)*(ϕ/β))
end

order  = 1
degree = 4

n      = 24
T      = 2*π*rₑ
nstep  = 4*n*(order+1)*4
dt     = T/nstep

model = CubedSphereDiscreteModel(n; radius=rₑ)

qf = advect_solid_body(model, order, degree,
                       q₀, u₀, dt, 0.5*dt, nstep;
                       write_solution=true,
                       write_solution_freq=nstep/24)
end
