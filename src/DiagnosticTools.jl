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
  t = CSV.read(csv_file_path, Table)
  getproperty(t, fieldname)
end
