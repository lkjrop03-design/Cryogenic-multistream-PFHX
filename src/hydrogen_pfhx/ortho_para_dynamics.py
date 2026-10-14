"""Ortho-para hydrogen physics used by the catalysed streams: the equilibrium
composition (rotational partition functions, as tabulated by Woolley, Scott and
Brickwedde, 1948, Table 12), and the conversion rate (Donaubauer et al., 2019) that drives the species
and energy balances in bvp_model.bvp_function.
"""

import numpy as np


# H2 rotational levels E_J (in K) of the vibrational ground state, J = 0 ... 15:
# E_J = (B0 J(J+1) - D0 J^2 (J+1)^2) hc/k with B0 = 59.322 cm-1, D0 = 0.0471 cm-1
_HC_K = 1.438776877
_ROT_LEVELS = [(J, (59.322 * J * (J + 1) - 0.0471 * (J * (J + 1))**2) * _HC_K)
               for J in range(16)]
# gas constant, J/(mol K)
_R_GAS = 8.31446261815324


def para_ortho_equilibrium(T):
    """Equilibrium para and ortho fractions at temperature T (K), from the
    rotational partition functions: para occupies even J (spin weight 1),
    ortho odd J (spin weight 3). Works on floats, arrays and dual numbers.
    """
    para = 0.0
    ortho = 0.0
    for J, E in _ROT_LEVELS:
        # Boltzmann weight of level J, summed separately for even (para) and odd (ortho) J
        w = (2 * J + 1) * np.exp(-E / T)
        if J % 2 == 0:
            para = para + w
        else:
            ortho = ortho + 3 * w
    # equilibrium para fraction = para share of the total partition function
    xp = para / (para + ortho)
    return xp, 1 - xp


# J=0 -> J=1 energy gap (~1417 J/mol): scale of the conversion heat, used by continuation.py
ORTHO_PARA_GROUND_STATE_GAP = _R_GAS * (_ROT_LEVELS[1][1] - _ROT_LEVELS[0][1])


# lowest molar density (mol/m3) at which the rate fit is evaluated; its denominator
# a*c + b vanishes at 15.96 mol/m3. Flagged in outputs.check_physical_consistency
KINETICS_MIN_MOLAR_DENSITY = 20.0


def first_order_kinetics(reactant):
    """Conversion rate (mol para per m3 of packed duct per s) of a catalysed
    hydrogen stream, first-order model of Donaubauer et al. (2019), Eqs. (4)-(6)
    with the averaged coefficients of their Table 2 (and the published
    erratum: Arrhenius exponent -Ea/(R T); Table 2's [x10^3 ...] means the
    listed number is the quantity x 10^3). Positive when para is produced.
    """
    # local state of the reacting stream: temperature, ortho fractions, concentration
    T = reactant.temperature
    xo = reactant.get_xo()
    xo_equil = reactant.get_xo_equil()
    cH2 = reactant.molar_density

    R = 8.31446261815324

    # averaged first-order coefficients (Donaubauer Table 2): Ea in J/mol, a in m3 s/mol, b in s
    Ea = -336.45
    a = 2.2e-3
    b = -35.11e-3

    # keep the denominator positive below the fit's range (c_value: plain value, also for dual numbers)
    c_value = getattr(cH2, 'val', cH2)
    c_den = cH2 if c_value >= KINETICS_MIN_MOLAR_DENSITY else KINETICS_MIN_MOLAR_DENSITY
    # rate constant k_first (1/s), Eq. (6)
    k = np.exp(-Ea/(R*T))/(a*c_den + b) * 1/(1-xo_equil)

    # first-order rate, Eq. (4): proportional to the distance from equilibrium
    conversion_rate = k * cH2 * (xo - xo_equil)

    return conversion_rate


