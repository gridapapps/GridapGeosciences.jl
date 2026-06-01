ambient_sgrad(f::Function) = m -> ambient_sgrad(f,m)
function ambient_sgrad(f::Function,forward_map::Field)
  function _gradf(x)
    inverse_map  = CubedSphereInverseMap(forward_map)
    αβ = inverse_map(x)
    sgrad(panel_to_cartesian(f),forward_map)(αβ)
  end
end

ambient_surflap(f::Function) = m -> ambient_surflap(f,m)
function ambient_surflap(f::Function,forward_map::Field)
  function _lapf(x)
    inverse_map  = CubedSphereInverseMap(forward_map)
    αβ = inverse_map(x)
    surflap(panel_to_cartesian(f),forward_map)(αβ)
  end
end

#### WARNING! Here we pullback in contravariant components. i.e. use contra_v
#### This is not applicable to Hdiv fields, which pull with Piola map
#### If computing ambient_surfdiv of vec ∈ Hdiv(S), replace contra_v with piola
ambient_surfdiv(f::Function) = m -> ambient_surfdiv(f,m)
function ambient_surfdiv(f::Function,forward_map::Field)
  function _sdivf(x)
    inverse_map  = CubedSphereInverseMap(forward_map)
    αβ = inverse_map(x)
    surfdiv(contra_v(panel_to_cartesian(f)),forward_map)(αβ)
  end
end
