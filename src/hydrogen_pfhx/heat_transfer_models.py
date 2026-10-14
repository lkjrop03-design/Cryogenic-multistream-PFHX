"""Material and fin correlations used by hexs.PlateFinHex: the aluminium fin
conductivity (fin efficiency) and the Manglik-Bergles offset-strip fin
correlation (film coefficient and friction factor of unpacked streams).
"""

import numpy as np

# coefficients a_0 ... a_8 of the NIST aluminium 3003-F conductivity fit
_AL3003 = (0.63736, -1.1437, 7.4624, -12.6905, 11.9165, -6.18721, 1.63939, -0.172667, 0.0)
_LN10 = np.log(10.0)


def _al3003(T):
    """NIST fit for aluminium 3003-F, valid 4-300 K: log10(k) = sum a_i (log10 T)^i."""
    # evaluate the polynomial in log10(T) with log/exp only (dual-number safe)
    x = np.log(T) / _LN10
    total = 0.0
    for i, a in enumerate(_AL3003):
        total = total + a * x**i
    return np.exp(total * _LN10)


def aluminium_thermal_conductivity(temperature):
    """Fin and parting-sheet conductivity (W/(m K)) at temperature T (K).
    Outside 4-300 K the fit is not extrapolated: held constant above 300 K and
    proportional to T below 4 K. Works on floats, arrays and dual numbers.
    """
    # plain value of T, also when T is a dual number
    value = getattr(temperature, 'val', temperature)
    # scalar (or dual) temperature: clamp the fit to its validity range
    if np.ndim(value) == 0:
        if value > 300.0:
            return _al3003(300.0)
        if value < 4.0:
            return _al3003(4.0) * temperature / 4.0
        return _al3003(temperature)
    # array of temperatures: same clamping, element-wise
    T = np.clip(temperature, 4.0, 300.0)
    k = _al3003(T)
    return np.where(temperature < 4.0, k * temperature / 4.0, k)


def manglik_bergles_heat_transfer_model(Re, alpha, delta, gamma):
    """Fanning friction factor f and Colburn j factor of rectangular
    offset-strip fins (Manglik & Bergles, 1995; fitted for 120 < Re < 1e4).
    alpha = s/h, delta = t/l, gamma = t/s; Re must use the same paper's
    hydraulic diameter (hexs.offset_strip_hydraulic_diameter).
    """
    # friction factor and j factor (Manglik & Bergles 1995, Eqs. for f and j)
    f = 9.6243 * Re**-0.7422 * alpha**-0.1856 * delta**0.3053 * gamma**-0.2659 * \
        (1 + 7.669e-8 * Re**4.429 * alpha**0.920 * delta**3.767 * gamma**0.236)**0.1
    j = 0.6522 * Re**-0.5403 * alpha**-0.1541 * delta**0.1499 * gamma**-0.0678 * \
        (1 + 5.269e-5 * Re**1.340 * alpha**0.504 * delta**0.456 * gamma**-1.055)**0.1
    return f, j
