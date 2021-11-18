"""
  Kinetic energy
"""
function Eₖ(uh,h,dΩ)
  0.5*sum(∫(uh⋅uh*h)dΩ)
end

"""
  Potential energy
"""
function Eₚ(a,b,c,dΩ)
  c*sum(∫(a*b)dΩ)
end

"""
  Total energy
"""
function Eₜ(uh,H,hh,g,dΩ)
  Eₖ(uh,H,dΩ)+Eₚ(hh,hh,0.5*g,dΩ)
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

"""Write scalar diagnostics to csv.
Diagnostics should be added as kwargs: field=values"""
function write_to_csv(csv_file_path; kwargs...)
  df = DataFrame(kwargs... )
  header = names(df)
  CSV.write(csv_file_path, df, header=header)
end

"""Append line to existing csv file.
row should be given as kwargs: field=value"""
function append_to_csv(csv_file_path; kwargs...)
  df = DataFrame(kwargs )
  CSV.write(csv_file_path, df, delim=",", append=true)
end

"""Create a .cvs file header. header should be specified as args"""
function initialize_csv(csv_file_path, args...)
  header = [String(_) for _ in args]
  CSV.write(csv_file_path,[], writeheader=true, header=header)
end

"""Wrapper to get a vector from a csv field. The fieldname argument should be passed as a symbol,
i.e. :<fieldname>"""
function get_scalar_field_from_csv(csv_file_path, fieldname)
  t = CSV.read(csv_file_path, DataFrame)
  getproperty(t, fieldname)
end

"""
  Full diagnostics for the shallow water equations (mass, vorticity, kinetic energy, potential energy, power)
"""

function compute_diagnostics_shallow_water!(h_tmp, w_tmp,
                                         model, dΩ, dω, S, L2MM, H1MM,
					 h, u, w, ϕ, F, h2, c)
  mass_i = compute_total_mass!(h_tmp, L2MM, get_free_dof_values(h))
  vort_i = compute_total_mass!(w_tmp, H1MM, get_free_dof_values(w))
  kin_i  = Eₖ(u,h,dΩ)
  pot_i  = Eₚ(h,h2,c,dΩ)
  pow_i  = sum(∫(ϕ*DIV(F))dω)

  mass_i, vort_i, kin_i, pot_i, pow_i
end

function dump_diagnostics_shallow_water!(h_tmp, w_tmp,
                                         model, dΩ, dω, S, L2MM, H1MM,
                                         h, u, w, ϕ, F, g, step, dt,
                                         output_file, dump_on_screen)

  mass_i, vort_i, kin_i, pot_i, pow_i = compute_diagnostics_shallow_water!(
                                          h_tmp, w_tmp,
                                          model, dΩ, dω, S, L2MM, H1MM,
                                          h, u, w, ϕ, F, h, 0.5*g)

  append_to_csv(output_file;
                time       = step*dt/24/60/60,
                mass       = mass_i,
                vorticity  = vort_i,
                kinetic    = kin_i,
                potential  = pot_i,
                power      = pow_i)

  if dump_on_screen
    @printf("%5d %14.9e %14.9e %14.9e %14.9e %14.9e %14.9e\n",
             step, mass_i, vort_i, kin_i, pot_i, kin_i+pot_i, pow_i)
  end
end

function dump_diagnostics_thermal_shallow_water!(h_tmp, w_tmp,
                                         model, dΩ, dω, S, L2MM, H1MM,
                                         h, u, E, w, ϕ, F, eF, step, dt,
                                         output_file, dump_on_screen)

  mass_i, vort_i, kin_i, pot_i, pow_k2p_i = compute_diagnostics_shallow_water!(
                                              h_tmp, w_tmp,
                                              model, dΩ, dω, S, L2MM, H1MM,
                                              h, u, w, ϕ, F, E, 0.5)

  buoy_i    = compute_total_mass!(h_tmp, L2MM, get_free_dof_values(E))
  pow_k2i_i = 0.5*sum(∫(h*DIV(eF))dω)

  append_to_csv(output_file;
                time       = step*dt/24/60/60,
                mass       = mass_i,
                vorticity  = vort_i,
                buoyancy   = buoy_i,
                kinetic    = kin_i,
                internal   = pot_i,
                power_k2p  = pow_k2p_i,
                power_k2i  = pow_k2i_i)

  if dump_on_screen
    @printf("%5d %14.9e %14.9e %14.9e %14.9e %14.9e %14.9e %14.9e %14.9e\n",
             step, mass_i, vort_i, buoy_i, kin_i, pot_i, kin_i+pot_i, pow_k2p_i, pow_k2i_i)
  end
end

function dump_diagnostics_thermal_shallow_water_mat_adv!(h_tmp, w_tmp,
                                         model, dΩ, dω, S, L2MM, H1MM,
                                         h, u, e, w, ϕ, F, de, step, dt,
                                         output_file, dump_on_screen)

  mass_i    = compute_total_mass!(h_tmp, L2MM, get_free_dof_values(h))
  vort_i    = compute_total_mass!(w_tmp, H1MM, get_free_dof_values(w))
  buoy_i    = sum(∫(h*e)dΩ)
  kin_i     = Eₖ(u,h,dΩ)
  pot_i     = Eₚ(h*h,e,0.5,dΩ)
  pow_k2p_i = sum(∫(ϕ*DIV(F))dω)
  pow_k2i_i = -0.5*sum(∫(F⋅de*h*h)dΩ)

  append_to_csv(output_file;
                time       = step*dt/24/60/60,
                mass       = mass_i,
                vorticity  = vort_i,
                buoyancy   = buoy_i,
                kinetic    = kin_i,
                internal   = pot_i,
                power_k2p  = pow_k2p_i,
                power_k2i  = pow_k2i_i)

  if dump_on_screen
    @printf("%5d %14.9e %14.9e %14.9e %14.9e %14.9e %14.9e %14.9e %14.9e\n",
             step, mass_i, vort_i, buoy_i, kin_i, pot_i, kin_i+pot_i, pow_k2p_i, pow_k2i_i)
  end
end

function dump_diagnostics_thermal_shallow_water_flux_adv!(h_tmp, w_tmp,
                                         model, dΩ, dω, S, L2MM, H1MM,
                                         h, u, E, w, ϕ, F, eF, step, dt,
                                         output_file, dump_on_screen)

  mass_i, vort_i, kin_i, pot_i, pow_k2p_i = compute_diagnostics_shallow_water!(
                                              h_tmp, w_tmp,
                                              model, dΩ, dω, S, L2MM, H1MM,
                                              h, u, w, ϕ, F, E, 0.5)

  buoy_i    = compute_total_mass!(w_tmp, H1MM, get_free_dof_values(E))
  pow_k2i_i = 0.5*sum(∫(h*DIV(eF))dω)

  append_to_csv(output_file;
                time       = step*dt/24/60/60,
                mass       = mass_i,
                vorticity  = vort_i,
                buoyancy   = buoy_i,
                kinetic    = kin_i,
                internal   = pot_i,
                power_k2p  = pow_k2p_i,
                power_k2i  = pow_k2i_i)

  if dump_on_screen
    @printf("%5d %14.9e %14.9e %14.9e %14.9e %14.9e %14.9e %14.9e %14.9e\n",
             step, mass_i, vort_i, buoy_i, kin_i, pot_i, kin_i+pot_i, pow_k2p_i, pow_k2i_i)
  end
end
