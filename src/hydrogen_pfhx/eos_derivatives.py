"""Analytic derivatives of the hydrogen properties with respect to density and
temperature, for the Leachman et al. (2009) equations of state as implemented
in hydrogen_thermodynamics.py (ortho, para and normal hydrogen).

hydrogen_property_derivatives.py combines these into derivatives of the
mixture properties with respect to the solver's state (T, P, xp), which
dualize.py seeds into dual numbers for the analytic Jacobian. Each function
reproduces finite differences of the corresponding hydrogen_thermodynamics
method; third-order Helmholtz terms are derived symbolically with sympy.
"""

import numpy as np
import sympy as sp

# ------------------------------------------------------------------------
# Third-order derivatives of the residual Helmholtz energy, needed for dcp.
# The residual part is a sum of polynomial, exponentially damped and Gaussian
# terms (Leachman et al., Eq. 32); for the last two, the derivative factors
# relative to the term itself are derived symbolically once, at import.
# ------------------------------------------------------------------------
_delta, _tau, _d, _t, _p, _phi, _beta, _gamma, _D = sp.symbols(
    'delta tau d t p phi beta gamma D', real=True, positive=True)


def _extra_factor(term_expr, nd, nt):
    """delta^nd tau^nt d^(nd+nt)(term)/d(delta)^nd d(tau)^nt divided by the term."""
    expr = term_expr
    for _ in range(nd):
        expr = sp.diff(expr, _delta)
    for _ in range(nt):
        expr = sp.diff(expr, _tau)
    expr = expr * _delta**nd * _tau**nt
    return sp.simplify(expr / term_expr)


# the four third-order derivatives: (order in delta, order in tau, name)
_targets = [(3, 0, 'ddd'), (2, 1, 'ddt'), (1, 2, 'dtt'), (0, 3, 'ttt')]

# exponentially damped terms: factor functions of (delta, tau, d, t, p)
_term2 = _delta**_d * _tau**_t * sp.exp(-_delta**_p)
_b2_factors = {name: _extra_factor(_term2, nd, nt) for nd, nt, name in _targets}
_b2_funcs = {name: sp.lambdify((_delta, _tau, _d, _t, _p), expr, 'numpy')
             for name, expr in _b2_factors.items()}

# Gaussian terms: factor functions of (delta, tau, d, t, phi, beta, gamma, D)
_g3 = _phi * (_delta - _D)**2 + _beta * (_tau - _gamma)**2
_term3 = _delta**_d * _tau**_t * sp.exp(-_g3)
_b3_factors = {name: _extra_factor(_term3, nd, nt) for nd, nt, name in _targets}
_b3_funcs = {name: sp.lambdify((_delta, _tau, _d, _t, _phi, _beta, _gamma, _D), expr, 'numpy')
             for name, expr in _b3_factors.items()}


def alphar_3rd_order(rho, T, sp_obj, which):
    """Third-order residual Helmholtz derivative `which` ('ddd', 'ddt', 'dtt' or
    'ttt'), pre-multiplied by delta^nd tau^nt like the first- and second-order
    methods of hydrogen_thermodynamics. sp_obj is the isomer's EOS object.
    """
    delta_v = rho / sp_obj.rho_c
    tau_v = sp_obj.Tc / T
    l, m, n = sp_obj.l, sp_obj.m, sp_obj.n
    Ni = np.array(sp_obj.Ni_eos)
    di = np.array(sp_obj.di)
    ti = np.array(sp_obj.ti)
    pi_ = np.array(sp_obj.pi)
    phi_a = np.array(sp_obj.phi_i)
    beta_a = np.array(sp_obj.beta_i)
    gamma_a = np.array(sp_obj.gamma_i)
    D_a = np.array(sp_obj.D_i)
    main_terms = Ni * delta_v**di * tau_v**ti

    # polynomial terms: the derivative factor is a product of the exponents
    b1 = {
        'ddd': di * (di - 1) * (di - 2),
        'ddt': di * ti * (di - 1),
        'dtt': di * ti * (ti - 1),
        'ttt': ti * (ti - 1) * (ti - 2),
    }[which][:l]

    # sum the three term families (polynomial, damped, Gaussian)
    total = np.sum(b1 * main_terms[:l])
    total += np.sum(main_terms[l:m] * np.exp(-(delta_v)**pi_[l:m]) *
                     _b2_funcs[which](delta_v, tau_v, di[l:m], ti[l:m], pi_[l:m]))
    g = phi_a * (delta_v - D_a)**2 + beta_a * (tau_v - gamma_a)**2
    total += np.sum(main_terms[m:n] * np.exp(-g) *
                     _b3_funcs[which](delta_v, tau_v, di[m:n], ti[m:n], phi_a, beta_a, gamma_a, D_a))
    return float(total)


def tau3_diff3_alpha0_difftau3(T, ak, bk, Tc):
    """tau^3 d^3(alpha_0)/d(tau)^3 of the ideal-gas Helmholtz energy (for dcp/dT)."""
    tau = Tc / T
    eff_bk = np.array(bk[2:])
    u = np.exp(eff_bk * tau)
    term = ak[2:] * eff_bk**3 * u * (u + 1) / (u - 1)**3
    return 3.0 + tau**3 * np.sum(term)


# ------------------------------------------------------------------------
# Density at constant P or T, from the implicit function theorem on P(rho, T)
# ------------------------------------------------------------------------
def dP_dT_rho(sp_obj, rho, T):
    """(dP/dT) at constant density, Pa/K."""
    deltaD1 = sp_obj.delta_diff_alphar_delta(rho, T)
    deltatauD1T = sp_obj.delta_tau_diff2_alphar_diff_delta_tau(rho, T)
    return rho * sp_obj.R * (1 + deltaD1 - deltatauD1T)


def drho_dT_P(sp_obj, rho, T):
    """(d rho/dT) at constant P = -(dP/dT)_rho / (dP/d rho)_T."""
    return -dP_dT_rho(sp_obj, rho, T) / sp_obj.diff_pressure_rho(rho, T)


def drho_dP_T(sp_obj, rho, T):
    """(d rho/dP) at constant T = 1 / (dP/d rho)_T."""
    return 1.0 / sp_obj.diff_pressure_rho(rho, T)


# ------------------------------------------------------------------------
# Molar enthalpy h = R T (1 + A), A = tau a0_tau + tau ar_tau + delta ar_delta;
# its derivatives need only first- and second-order Helmholtz terms
# ------------------------------------------------------------------------
def denthalpy_dT_rho(sp_obj, rho, T):
    """(dh/dT) at constant density, J/(mol K)."""
    tau_a0_tau = sp_obj.tau_diff_alpha0_tau(T)
    tau2_a0_tautau = sp_obj.tau2_diff2_alpha0_difftau2(T)
    # dA/dT, term by term (d tau/dT = -tau/T)
    term1 = -(1.0 / T) * (tau_a0_tau + tau2_a0_tautau)

    tau_ar_tau = sp_obj.tau_diff_alphar_tau(rho, T)
    tau2_ar_tautau = sp_obj.tau2_diff2_alphar_difftau2(rho, T)
    term2 = -(1.0 / T) * (tau_ar_tau + tau2_ar_tautau)

    delta_tau_ar_deltatau = sp_obj.delta_tau_diff2_alphar_diff_delta_tau(rho, T)
    term3 = -(1.0 / T) * delta_tau_ar_deltatau

    A = tau_a0_tau + tau_ar_tau + sp_obj.delta_diff_alphar_delta(rho, T)
    dA_dT = term1 + term2 + term3
    return sp_obj.R * (1 + A) + sp_obj.R * T * dA_dT


def denthalpy_drho_T(sp_obj, rho, T):
    """(dh/d rho) at constant T, J m3/mol2."""
    delta = rho / sp_obj.rho_c
    d_tauartau_ddelta = sp_obj.delta_tau_diff2_alphar_diff_delta_tau(rho, T) / delta
    deltaardelta = sp_obj.delta_diff_alphar_delta(rho, T)
    delta2_ardeltadelta = sp_obj.delta2_diff2_alphar_diffdelta2(rho, T)
    d_deltaardelta_ddelta = deltaardelta / delta + delta2_ardeltadelta / delta
    dA_ddelta = d_tauartau_ddelta + d_deltaardelta_ddelta
    return sp_obj.R * T * (dA_ddelta / sp_obj.rho_c)


# ------------------------------------------------------------------------
# Isobaric heat capacity cp = cv + R (1 + X1 - X1T)^2 / (1 + 2 X1 + X2), with
# X1 = delta ar_delta, X2 = delta^2 ar_deltadelta, X1T = delta tau ar_deltatau;
# its derivatives need the third-order Helmholtz terms above
# ------------------------------------------------------------------------
def dcp_dT_rho(sp_obj, rho, T):
    """(dcp/dT) at constant density, J/(mol K2)."""
    X1 = sp_obj.delta_diff_alphar_delta(rho, T)
    X2 = sp_obj.delta2_diff2_alphar_diffdelta2(rho, T)
    X1T = sp_obj.delta_tau_diff2_alphar_diff_delta_tau(rho, T)
    Y2_0 = sp_obj.tau2_diff2_alpha0_difftau2(T)
    Y2_r = sp_obj.tau2_diff2_alphar_difftau2(rho, T)

    ddt = alphar_3rd_order(rho, T, sp_obj, 'ddt')
    dtt = alphar_3rd_order(rho, T, sp_obj, 'dtt')
    ttt_r = alphar_3rd_order(rho, T, sp_obj, 'ttt')
    ttt_0 = tau3_diff3_alpha0_difftau3(T, sp_obj.ak, sp_obj.bk, sp_obj.Tc)

    # cv part: derivatives of the tau^2-weighted second derivatives
    dY20_dT = -(1.0 / T) * (2 * Y2_0 + ttt_0)
    dY2r_dT = -(1.0 / T) * (2 * Y2_r + ttt_r)
    dcv_dT = -sp_obj.R * (dY20_dT + dY2r_dT)

    # derivatives of X1, X1T, X2; then the quotient rule for the cp - cv term
    dX1_dT = -X1T / T
    dX1T_dT = -(1.0 / T) * (X1T + dtt)
    dX2_dT = -(1.0 / T) * ddt

    numerator = 1 + X1 - X1T
    denominator = 1 + 2 * X1 + X2
    dnum_dT = dX1_dT - dX1T_dT
    dden_dT = 2 * dX1_dT + dX2_dT

    return dcv_dT + sp_obj.R * (2 * numerator * dnum_dT * denominator - numerator**2 * dden_dT) / denominator**2


def dcp_drho_T(sp_obj, rho, T):
    """(dcp/d rho) at constant T, J m3/(mol2 K)."""
    delta = rho / sp_obj.rho_c
    rho_c = sp_obj.rho_c
    X1 = sp_obj.delta_diff_alphar_delta(rho, T)
    X2 = sp_obj.delta2_diff2_alphar_diffdelta2(rho, T)
    X1T = sp_obj.delta_tau_diff2_alphar_diff_delta_tau(rho, T)

    ddd = alphar_3rd_order(rho, T, sp_obj, 'ddd')
    ddt = alphar_3rd_order(rho, T, sp_obj, 'ddt')
    dtt = alphar_3rd_order(rho, T, sp_obj, 'dtt')

    # cv part, derivatives of X1, X1T, X2, then the quotient rule
    dcv_drho = -sp_obj.R * (1.0 / rho_c) * (dtt / delta)

    dX1_drho = (1.0 / rho_c) * (X1 + X2) / delta
    dX1T_drho = (1.0 / (rho_c * delta)) * (X1T + ddt)
    dX2_drho = (1.0 / (rho_c * delta)) * (2 * X2 + ddd)

    numerator = 1 + X1 - X1T
    denominator = 1 + 2 * X1 + X2
    dnum_drho = dX1_drho - dX1T_drho
    dden_drho = 2 * dX1_drho + dX2_drho

    return dcv_drho + sp_obj.R * (2 * numerator * dnum_drho * denominator - numerator**2 * dden_drho) / denominator**2


# ------------------------------------------------------------------------
# Viscosity of normal hydrogen (Muzny et al. correlation, as in
# hydrogen_thermodynamics): built symbolically once and differentiated
# ------------------------------------------------------------------------
_rho_s, _T_s, _M_s = sp.symbols('rho T M', positive=True)
_v_ai = [2.09630e-1, -4.55274e-1, 1.43602e-1, -3.35325e-2, 2.76981e-3]
_v_bi = [-0.187, 2.4871, 3.7151, -11.0972, 9.0965, -3.8292, 0.5166]
_v_ci = [6.43449673, 4.56334068e-2, 2.32797868e-1, 9.58326120e-1, 1.27941189e-1, 3.63576595e-1]

# dilute-gas, initial-density and higher-density contributions
_omega_v = sp.Float(0.297)
_e_kb = sp.Float(30.41)
_T_star = _T_s / _e_kb
_S_star = sp.exp(sum(_v_ai[i] * sp.log(_T_star)**i for i in range(len(_v_ai))))
_mu_0 = sp.Float(0.021357) * (_M_s * 1000 * _T_s)**sp.Rational(1, 2) / (_omega_v**2 * _S_star) * sp.Float(1e-6)
_B_mu_star = sum(_v_bi[i] * _T_star**(-i) for i in range(len(_v_bi)))
_B_mu = _B_mu_star * _omega_v**3
_mu_1 = _B_mu * _mu_0
_mu_init = _mu_1 * _rho_s * _M_s
_rho_sc = sp.Float(90.5)
_Tc_fixed = sp.Float(33.145)
_rho_r = _rho_s * _M_s / _rho_sc
_T_r = _T_s / _Tc_fixed
_mu_add = sp.Float(1e-6) * _v_ci[0] * _rho_r**2 * sp.exp(
    _v_ci[1] * _T_r + _v_ci[2] / _T_r + _v_ci[3] * _rho_r**2 / (_v_ci[4] + _T_r) + _v_ci[5] * _rho_r**6)
_mu_expr = _mu_0 + _mu_init + _mu_add

_dmu_dT_func = sp.lambdify((_rho_s, _T_s, _M_s), sp.diff(_mu_expr, _T_s), 'numpy')
_dmu_drho_func = sp.lambdify((_rho_s, _T_s, _M_s), sp.diff(_mu_expr, _rho_s), 'numpy')


def dviscosity_dT_rho(rho, T, M):
    """(d mu/dT) at constant molar density rho; M is the molar mass (kg/mol)."""
    return float(_dmu_dT_func(rho, T, M))


def dviscosity_drho_T(rho, T, M):
    """(d mu/d rho) at constant T."""
    return float(_dmu_drho_func(rho, T, M))


# ------------------------------------------------------------------------
# Thermal conductivity: derivatives of the dilute-gas and excess parts only.
# The critical enhancement is left out, so these derivatives are approximate
# within a few K of hydrogen's critical temperature (Jacobian accuracy only;
# the solved properties use the full correlation).
# ------------------------------------------------------------------------
def _build_k_derivs(sp_obj):
    """Symbolic (dk/dT, dk/d rho) functions for the isomer of sp_obj."""
    Tc, rho_c = sp_obj.Tc, sp_obj.rho_c
    T_r = _T_s / Tc
    rho_r = _rho_s / rho_c
    lambda_0 = sum(sp_obj.tc_A1[i] * T_r**i for i in range(len(sp_obj.tc_A1))) / \
        sum(sp_obj.tc_A2[j] * T_r**j for j in range(len(sp_obj.tc_A2)))
    lambda_excess = sum((sp_obj.tc_B1[b] + sp_obj.tc_B2[b] * T_r) * rho_r**b
                         for b in range(len(sp_obj.tc_B1)))
    k_expr = lambda_0 + lambda_excess
    return (sp.lambdify((_rho_s, _T_s), sp.diff(k_expr, _T_s), 'numpy'),
            sp.lambdify((_rho_s, _T_s), sp.diff(k_expr, _rho_s), 'numpy'))


# derivative functions per isomer class (the coefficients are fixed per isomer)
_k_deriv_cache = {}


def dthermal_conductivity_dT_rho(sp_obj, rho, T):
    """(dk/dT) at constant density, W/(m K2)."""
    key = type(sp_obj).__name__
    if key not in _k_deriv_cache:
        _k_deriv_cache[key] = _build_k_derivs(sp_obj)
    dk_dT_func, _ = _k_deriv_cache[key]
    return float(dk_dT_func(rho, T))


def dthermal_conductivity_drho_T(sp_obj, rho, T):
    """(dk/d rho) at constant T."""
    key = type(sp_obj).__name__
    if key not in _k_deriv_cache:
        _k_deriv_cache[key] = _build_k_derivs(sp_obj)
    _, dk_drho_func = _k_deriv_cache[key]
    return float(dk_drho_func(rho, T))
