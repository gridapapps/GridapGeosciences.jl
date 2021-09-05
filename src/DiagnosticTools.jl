"""
  Kinetic energy
"""
function Eₖ(uh,H,dΩ)
  0.5*H*sum(∫(uh⋅uh)dΩ)
end

"""
  Potential energy
"""
function Eₚ(hh,g,dΩ)
  0.5*g*sum(∫(hh*hh)dΩ)
end

"""
  Total energy
"""
function Eₜ(uh,H,hh,g,dΩ)
  Eₖ(uh,H,dΩ)+Eₚ(hh,g,dΩ)
end

"""
  Kinetic to potential
"""
function compute_kin_to_pot!(w,unv,divvh,hnv)
  mul!(w,divvh,hnv)
  unv⋅w
end

"""
  Potential to kinetic
"""
function compute_pot_to_kin!(w,hnv,qdivu,unv)
   mul!(w,qdivu,unv)
   hnv⋅w
end

"""
  Total Mass
"""
function compute_total_mass!(w,L2MM,hh)
  mul!(w,L2MM,hh)
  sum(w)
end

"""
  Full diagnostics for the shallow water equations (mass, vorticity, kinetic energy, potential energy, power)
"""
function compute_diagnostics_shallow_water!(model, dΩ, dω, S, L2MM, H1MM, H1MMchol, h_tmp, w_tmp, g, h, u, ϕ, F, step, to_std, out_dir, w)
  mass_i = compute_total_mass!(h_tmp, L2MM, get_free_dof_values(h))
  # diagnose the vorticity
  n    = get_normal_vector(model)
  a(s) = ∫(perp(n,∇(s))⋅(u))dΩ
  rhs  = assemble_vector(a, S)
  copy!(get_free_dof_values(w), rhs)
  ldiv!(H1MMchol, get_free_dof_values(w))
  vort_i = compute_total_mass!(w_tmp, H1MM, get_free_dof_values(w))
  kin_i  = 0.5*sum(∫(h*(u⋅u))dΩ)
  pot_i  = 0.5*g*sum(∫(h*h)dΩ)
  pow_i  = sum(∫(ϕ*DIV(F))dω)

  # save to file
  save(joinpath(out_dir,"swe_diagnostics.jld"), "mass", mass_i, "vort", vort_i, "kinetic", kin_i, "potential", pot_i, "power", pow_i)
  if to_std
    println(step, "\t", mass_i, "\t", vort_i, "\t", kin_i, "\t", pot_i, "\t", kin_i+pot_i, "\t", pow_i)
  end
end

