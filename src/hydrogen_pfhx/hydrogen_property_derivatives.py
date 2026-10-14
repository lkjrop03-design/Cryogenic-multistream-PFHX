"""Analytic derivatives of a hydrogen stream's mixture properties with respect
to its temperature, pressure and para fraction, built from the equation-of-
state derivatives in eos_derivatives and the same mixing rules as
fluids.Hydrogen. Used by dualize.dualize_stream for the analytic Jacobian.
"""


from hydrogen_pfhx import eos_derivatives as ed


def hydrogen_property_derivatives(h):
    """Derivatives of mass density, viscosity, cp, thermal conductivity and
    enthalpy of a hydrogen stream at its current state.
    Returns {property: {'dT': per K, 'dP': per Pa, 'dxp': per unit fraction}}.
    """
    ortho, para, normal = h.ortho, h.para, h.normal
    rho_o, rho_p, rho_n = h.ortho_density, h.para_density, h.normal_density
    T, xp, M = h.temperature, h.xp, h.molecular_mass

    # density derivatives of each form at constant P and T
    drho_o_dT = ed.drho_dT_P(ortho, rho_o, T); drho_o_dP = ed.drho_dP_T(ortho, rho_o, T)
    drho_p_dT = ed.drho_dT_P(para, rho_p, T);   drho_p_dP = ed.drho_dP_T(para, rho_p, T)
    drho_n_dT = ed.drho_dT_P(normal, rho_n, T); drho_n_dP = ed.drho_dP_T(normal, rho_n, T)

    # mixture density: molar-volume mixing, as fluids.Hydrogen
    rho_mix = 1.0 / ((1 - xp) / rho_o + xp / rho_p)
    dmolar_dT = rho_mix**2 * ((1 - xp) / rho_o**2 * drho_o_dT + xp / rho_p**2 * drho_p_dT)
    dmolar_dP = rho_mix**2 * ((1 - xp) / rho_o**2 * drho_o_dP + xp / rho_p**2 * drho_p_dP)
    dmolar_dxp = -rho_mix**2 * (1 / rho_p - 1 / rho_o)
    dmass_density = {'dT': dmolar_dT * M, 'dP': dmolar_dP * M, 'dxp': dmolar_dxp * M}

    # viscosity of normal hydrogen: direct T dependence plus through its density
    dmu_dT_direct = ed.dviscosity_dT_rho(rho_n, T, M)
    dmu_drho = ed.dviscosity_drho_T(rho_n, T, M)
    dviscosity = {'dT': dmu_dT_direct + dmu_drho * drho_n_dT,
                  'dP': dmu_drho * drho_n_dP, 'dxp': 0.0}

    # frozen cp: mole-fraction average of ortho and para
    cp_o = ortho.specific_heat_constant_pressure(rho_o, T)
    cp_p = para.specific_heat_constant_pressure(rho_p, T)
    dcpo_dT_direct = ed.dcp_dT_rho(ortho, rho_o, T); dcpo_drho = ed.dcp_drho_T(ortho, rho_o, T)
    dcpp_dT_direct = ed.dcp_dT_rho(para, rho_p, T);   dcpp_drho = ed.dcp_drho_T(para, rho_p, T)
    dcpo_dT = dcpo_dT_direct + dcpo_drho * drho_o_dT; dcpo_dP = dcpo_drho * drho_o_dP
    dcpp_dT = dcpp_dT_direct + dcpp_drho * drho_p_dT; dcpp_dP = dcpp_drho * drho_p_dP
    dcp_molar_dT = (1 - xp) * dcpo_dT + xp * dcpp_dT
    dcp_molar_dP = (1 - xp) * dcpo_dP + xp * dcpp_dP
    dcp_molar_dxp = cp_p - cp_o
    dcp = {'dT': dcp_molar_dT / M, 'dP': dcp_molar_dP / M, 'dxp': dcp_molar_dxp / M}

    # conductivity: interpolation between normal and para hydrogen, as fluids.Hydrogen
    dk_p_dT_direct = ed.dthermal_conductivity_dT_rho(para, rho_p, T); dk_p_drho = ed.dthermal_conductivity_drho_T(para, rho_p, T)
    dk_n_dT_direct = ed.dthermal_conductivity_dT_rho(normal, rho_n, T); dk_n_drho = ed.dthermal_conductivity_drho_T(normal, rho_n, T)
    dph_k_dT = dk_p_dT_direct + dk_p_drho * drho_p_dT; dph_k_dP = dk_p_drho * drho_p_dP
    dnh_k_dT = dk_n_dT_direct + dk_n_drho * drho_n_dT; dnh_k_dP = dk_n_drho * drho_n_dP
    frac = (0.75 - (1 - xp)) / 0.75
    # background viscosity only enters the small critical enhancement
    mu_bg_dummy = 6e-6
    ph_k = para.thermal_conductivity(rho_p, T, mu_bg_dummy)[0]
    nh_k = normal.thermal_conductivity(rho_n, T, mu_bg_dummy)[0]
    dk_dT = dnh_k_dT + (dph_k_dT - dnh_k_dT) * frac
    dk_dP = dnh_k_dP + (dph_k_dP - dnh_k_dP) * frac
    dk_dxp = (ph_k - nh_k) * (1.0 / 0.75)
    dthermal_conductivity = {'dT': dk_dT, 'dP': dk_dP, 'dxp': dk_dxp}

    # enthalpy: mole-fraction average; d/dxp is the conversion enthalpy
    h_o = ortho.enthalpy(rho_o, T); h_p = para.enthalpy(rho_p, T)
    dho_dT_direct = ed.denthalpy_dT_rho(ortho, rho_o, T); dho_drho = ed.denthalpy_drho_T(ortho, rho_o, T)
    dhp_dT_direct = ed.denthalpy_dT_rho(para, rho_p, T);   dhp_drho = ed.denthalpy_drho_T(para, rho_p, T)
    dho_dT = dho_dT_direct + dho_drho * drho_o_dT; dho_dP = dho_drho * drho_o_dP
    dhp_dT = dhp_dT_direct + dhp_drho * drho_p_dT; dhp_dP = dhp_drho * drho_p_dP
    dh_molar_dT = (1 - xp) * dho_dT + xp * dhp_dT
    dh_molar_dP = (1 - xp) * dho_dP + xp * dhp_dP
    dh_molar_dxp = h_p - h_o
    denthalpy = {'dT': dh_molar_dT / M, 'dP': dh_molar_dP / M, 'dxp': dh_molar_dxp / M}

    return {'mass_density': dmass_density, 'viscosity': dviscosity,
            'specific_heat_capacity': dcp, 'thermal_conductivity': dthermal_conductivity,
            'enthalpy': denthalpy}
