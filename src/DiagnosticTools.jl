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
function compute_diagnostics_shallow_water!(model, order, Ω, dΩ, dω, qₖ, wₖ, U, V, R, S, L2MM, H1MM, g, h, u, ϕ, F, mass, vort, kin, pot, pow, step, do_print)
  mass_i = sum(L2MM*get_free_dof_values(h))
  w      = diagnose_vorticity(model, order, Ω, qₖ, wₖ, R, S, U, V, H1MM, u)
  vort_i = sum(H1MM*get_free_dof_values(w))
  kin_i  = 0.5*sum(∫(h*(u⋅u))dΩ)
  pot_i  = 0.5*g*sum(∫(h*h)dΩ)
  pow_i  = sum(∫(ϕ*DIV(F))dω)

  mass[step] = mass_i
  vort[step] = vort_i
  kin[step]  = kin_i
  pot[step]  = pot_i
  pow[step]  = pow_i

  if do_print
    # normalised conservation errors
    mass_norm = (mass_i-mass[1])/mass[1]
    vort_norm = vort_i-vort[1]
    en_norm   = (kin_i+pot_i-kin[1]-pot[1])/(kin[1]+pot[1])
    println(step, "\t", mass_norm, "\t", vort_norm, "\t", kin_i, "\t", pot_i, "\t", en_norm, "\t", pow_i)
  end

  w
end

