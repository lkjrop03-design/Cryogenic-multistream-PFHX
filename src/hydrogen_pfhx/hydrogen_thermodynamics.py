"""Leachman et al. (2009) equations of state for normal, ortho and para
hydrogen (reduced Helmholtz energy alpha = alpha_0 + alpha_r in delta = rho/rho_c
and tau = Tc/T), with the Muzny et al. viscosity and Assael et al. thermal
conductivity correlations. fluids.Hydrogen uses one object per isomer to
obtain densities, enthalpies, heat capacities and transport properties.
"""

import numpy as np
import scipy.optimize
import CoolProp.CoolProp as CP

class HydrogenData(object):

    """Shared equation-of-state machinery; the subclasses supply the
    coefficients of each hydrogen form.
    """
    # gas constant (J/(mol K)) and the term counts of the three alpha_r term groups
    R = 8.31446261815324

    l = 7
    m = 9
    n = 14


    # viscosity correlation coefficients (Muzny et al.)
    v_ai = [2.09630e-1,	-4.55274e-1,	1.43602e-1,	-3.35325e-2,	2.76981e-3]
    v_bi = [-0.187,	2.4871,	3.7151,	-11.0972,	9.0965,	-3.8292,	0.5166]
    v_ci = [6.43449673,	4.56334068e-2,	2.32797868e-1,	9.58326120e-1,	1.27941189e-1,	3.63576595e-1]


    def __init__(self):
        """Zero the enthalpy at the reference state 50 K, 1 kPa: ideal gas, with a
        single (gas) density root. Called at the end of each subclass's
        __init__, after its coefficients are set.
        """
        self.p_ref = 1000.0
        self.t_ref = 50.0
        rho_guess = self.p_ref / (self.R * self.t_ref)
        rho = self.calculate_density(self.p_ref, self.t_ref, rho_guess)
        self.h_ref = self.absolute_enthalpy(rho, self.t_ref)

    def pressure(self, rho, T):
        """Pressure (Pa) at molar density rho (mol/m3) and T (K)."""
        d_alphar_delta = self.delta_diff_alphar_delta(rho, T)
        p = rho*self.R*T * (1 + d_alphar_delta)
        return p

    def viscosity(self, rho, T, M):
        """Dynamic viscosity (Pa s) from Muzny et al.: dilute-gas, initial-density
        and higher-density contributions. M is the molar mass (kg/mol).
        """
        # dilute-gas viscosity from the effective collision integral
        omega = 0.297
        e_kb = 30.41
        T_star = T/e_kb
        S_star = np.exp(np.sum(self.v_ai*np.log(T_star)**np.arange(start=0,stop=len(self.v_ai))))
        mu_0 = 0.021357*(M*1e3*T)**0.5/(omega**2.0*S_star)*1e-6

        # initial-density contribution (second viscosity virial coefficient)
        B_mu_star = np.sum(self.v_bi*T_star**(-1*np.arange(start=0,stop=len(self.v_bi))))
        B_mu = B_mu_star * omega **3
        mu_1 = B_mu * mu_0
        mu_init = mu_1 * rho * M

        # higher-density contribution, in reduced density and temperature
        rho_sc = 90.5
        Tc = 33.145
        rho_r = rho * M / rho_sc
        T_r = T / Tc

        mu_add = 1e-6 * self.v_ci[0] * rho_r**2.0 * np.exp(self.v_ci[1]*T_r + self.v_ci[2]/T_r + self.v_ci[3]*rho_r**2/(self.v_ci[4] + T_r) + self.v_ci[5]*rho_r**6)

        mu = mu_0 + mu_init + mu_add
        return mu

    def enthalpy(self, rho, T):
        """Molar enthalpy (J/mol) relative to the reference state (h_ref)."""
        h = self.absolute_enthalpy(rho, T) - self.h_ref
        return h

    def entropy(self, rho, T):
        """Molar entropy (J/(mol K)) from the Helmholtz energy (no reference offset)."""
        s = self.R * (self.tau_diff_alpha0_tau(T) + self.tau_diff_alphar_tau(rho, T) - self.ideal_helmholtz_energy(rho, T) - self.residual_helmholtz_energy(rho, T))
        return s

    def absolute_enthalpy(self, rho, T):
        """Molar enthalpy (J/mol) of the equation of state before referencing."""
        h = self.R*T * (1 + self.tau_diff_alpha0_tau(T) + self.tau_diff_alphar_tau(rho, T) + self.delta_diff_alphar_delta(rho, T))
        return h

    def specific_heat_constant_volume(self, rho, T):
        """Molar cv (J/(mol K))."""
        tau2_d2_alpha0_dtau2 = self.tau2_diff2_alpha0_difftau2(T)
        tau2_d2_alphar_dtau2 = self.tau2_diff2_alphar_difftau2(rho, T)
        cv = -self.R * (tau2_d2_alpha0_dtau2 + tau2_d2_alphar_dtau2)
        return cv

    def specific_heat_constant_pressure(self, rho, T):
        """Molar cp (J/(mol K)) from cv and the density derivatives of alpha_r."""
        cv = self.specific_heat_constant_volume(rho, T)
        # cp = cv + R (1 + d alpha_r - d tau alpha_r)^2 / (1 + 2 d alpha_r + d2 alpha_r)
        numerator = 1 + self.delta_diff_alphar_delta(rho, T) - self.delta_tau_diff2_alphar_diff_delta_tau(rho,T)
        denominator = 1 + 2.0*self.delta_diff_alphar_delta(rho, T) + self.delta2_diff2_alphar_diffdelta2(rho,T)
        cp = cv + self.R*numerator**2/denominator
        return cp

    def pressure_error(self, rho, T, P):
        """Residual P(rho, T) - P, solved for rho by calculate_density."""
        pe = self.pressure(rho, T) - P
        return pe

    def calculate_density(self, P, T, rho_guess):
        """Molar density (mol/m3) at P (Pa) and T (K), by solving the equation of
        state from rho_guess. A result is accepted only if converged, positive and
        reproducing P to 1e-6; otherwise retried from an ideal-gas and a denser
        guess, then a RuntimeError is raised.
        """
        for guess in (rho_guess, P / (self.R * T), 3.0 * rho_guess):
            sol, _, ier, _ = scipy.optimize.fsolve(
                self.pressure_error, x0=guess, args=(T, P), xtol=1e-10, full_output=True)
            rho = sol[0]
            if ier == 1 and rho > 0 and abs(self.pressure(rho, T) - P) <= 1e-6 * P + 1e-3:
                return rho
        raise RuntimeError('%s: density solve failed at T = %.4g K, P = %.6g Pa'
                           % (type(self).__name__, T, P))

    def thermal_conductivity(self, rho, T, mu_background):
        """Thermal conductivity (W/(m K)) from Assael et al.: dilute-gas, excess
        and critical-enhancement (Olchowy-Sengers) parts. mu_background is the
        viscosity without its critical part. Returns (total, critical part).
        Not available for ortho-hydrogen (fluids.Hydrogen interpolates para/normal).
        """
        if isinstance(self, OrthoHydrogen):
            raise NotImplementedError('no thermal conductivity correlation for '
                                      'ortho-hydrogen; use the para/normal mixture')
        else:
            rho_r = rho / self.rho_c
            T_r = T / self.Tc

            ai_m = np.arange(len(self.tc_A1))
            ai_n = np.arange(len(self.tc_A2))
            # dilute-gas part: ratio of polynomials in reduced temperature
            lambda_0 = np.sum(self.tc_A1 * T_r**ai_m) / np.sum(self.tc_A2 * T_r**ai_n)

            bi = np.arange(len(self.tc_B1))
            # excess part: polynomial in reduced density
            lambda_excess = np.sum((self.tc_B1 + self.tc_B2 * T_r) * rho_r**bi)

            cp = self.specific_heat_constant_pressure(rho, T)
            cv = self.specific_heat_constant_volume(rho, T)

            # critical enhancement (Olchowy-Sengers): correlation length xi and crossover functions
            kb = 1.38064852e-23
            RD = 1.01
            nu = 0.63
            gamma = 1.2415
            GAMMA = 0.172
            xi_0 = 1.5e-10
            T_ref = 1.5 * self.Tc

            del_psi = self.R *self.Tc * rho / self.rho_c * (self.diff_rho_pressure(rho, T) - T_ref/T * self.diff_rho_pressure(rho, T_ref))
            min_val = 1e-16
            xi = xi_0 * (GAMMA**(-1) * np.max([min_val,del_psi])) ** (nu/gamma)
            y = self.qd*xi
            omega_0 = 2/np.pi * (1 - np.exp(-1/((y)**(-1.0) + (y*self.rho_c/rho)**2/3)))
            omega = 2/np.pi * ((cp - cv)/cp * np.arctan(y) + cv/cp*y)

            if xi > 0:
                lambda_c = rho * cp * RD * kb * T /(6*np.pi*mu_background*xi) * (omega - omega_0)
            else:
                lambda_c = 0

            # the enhancement is never negative
            lambda_c = np.max([0, lambda_c])


            lambda_val = lambda_0 + lambda_excess + lambda_c

        return lambda_val, lambda_c

    def ideal_helmholtz_energy(self, rho, T):
        """Ideal-gas part alpha_0 of the reduced Helmholtz energy."""
        delta = rho / self.rho_c
        tau = self.Tc / T
        alpha_0 = np.log(delta) + 1.5 * np.log(tau) + self.ak[0] + self.ak[1]*tau +\
            np.sum(self.ak[2:] * np.log(1 - np.exp(self.bk[2:]*tau)))
        return alpha_0

    def residual_helmholtz_energy(self, rho, T):
        """Residual part alpha_r: polynomial, exponential and Gaussian terms."""
        delta = rho / self.rho_c
        tau = self.Tc / T
        # alpha_r = sum of polynomial (first l), exponential (l..m) and Gaussian (m..n) terms
        main_terms = self.Ni_eos * (delta ** self.di) * (tau ** self.ti)
        gaussian_terms = self.phi_i *(delta - self.D_i)**2.0 + self.beta_i * (tau - self.gamma_i)**2
        alpha_r = np.sum(main_terms[:self.l]) + \
            np.sum(main_terms[self.l:self.m]*np.exp(-(delta) ** self.pi[self.l:self.m])) + \
            np.sum(main_terms[self.m:self.n]*np.exp(-gaussian_terms))
        return alpha_r

    def diff_rho_pressure(self, rho, T):
        """d(rho)/dP at constant T."""
        drho_dp = 1/self.diff_pressure_rho(rho, T)
        return drho_dp

    def diff_pressure_rho(self, rho, T):
        """dP/d(rho) at constant T."""
        dp_drho = self.R * T * (1 + 2.0*self.delta_diff_alphar_delta(rho, T) + self.delta2_diff2_alphar_diffdelta2(rho, T))
        return dp_drho

    def delta_diff_alphar_delta(self, rho, T):
        """delta * d(alpha_r)/d(delta)."""
        delta = rho / self.rho_c
        tau = self.Tc / T
        main_terms = self.Ni_eos * delta ** self.di * tau ** self.ti
        extra_terms = (self.di - self.pi * delta**self.pi)
        gaussian_terms = self.phi_i *(delta - self.D_i)**2.0 + self.beta_i * (tau - self.gamma_i)**2
        extra_gaussian_terms = self.di[self.m:self.n] - 2.0*delta*self.phi_i*(delta - self.D_i)
        d_alphar_delta = np.sum(self.di[:self.l]*main_terms[:self.l]) + \
            np.sum(main_terms[self.l:self.m]*np.exp(-(delta) ** self.pi[self.l:self.m])*extra_terms[self.l:self.m]) + \
            np.sum(main_terms[self.m:self.n]*np.exp(-gaussian_terms)*extra_gaussian_terms)
        return d_alphar_delta

    def tau_diff_alpha0_tau(self, T):
        """tau * d(alpha_0)/d(tau)."""
        tau = self.Tc / T
        d_alpha0_tau = 1.5 + tau*self.ak[1] +\
            tau*np.sum(self.ak[2:] * -1.0*self.bk[2:]*np.exp(self.bk[2:]*tau) / (1 - np.exp(self.bk[2:]*tau)))
        return d_alpha0_tau

    def tau_diff_alphar_tau(self, rho, T):
        """tau * d(alpha_r)/d(tau)."""
        delta = rho / self.rho_c
        tau = self.Tc / T
        main_terms = self.Ni_eos * delta ** self.di * tau ** self.ti
        extra_terms = self.ti[self.l:self.m]
        gaussian_terms = self.phi_i *(delta - self.D_i)**2.0 + self.beta_i * (tau - self.gamma_i)**2
        extra_gaussian_terms = self.ti[self.m:self.n] - 2.0*tau*self.beta_i*(tau - self.gamma_i)
        d_alphar_tau = np.sum(self.ti[:self.l]*main_terms[:self.l]) + \
            np.sum(main_terms[self.l:self.m]*np.exp(-(delta) ** self.pi[self.l:self.m])*extra_terms) + \
            np.sum(main_terms[self.m:self.n]*np.exp(-gaussian_terms)*extra_gaussian_terms)
        return d_alphar_tau

    def tau2_diff2_alpha0_difftau2(self, T):
        """tau^2 * d2(alpha_0)/d(tau)2."""
        tau = self.Tc / T
        eff_bk = np.array(self.bk[2:])
        d2_alpha0_dtau2 = -1.5 - tau**2.0 *np.sum(self.ak[2:] * eff_bk**2.0 * np.exp(eff_bk*tau) * ((np.exp(eff_bk*tau) - 1.0)**(-2.0)))
        return d2_alpha0_dtau2

    def tau2_diff2_alphar_difftau2(self, rho, T):
        """tau^2 * d2(alpha_r)/d(tau)2."""
        delta = rho / self.rho_c
        tau = self.Tc / T
        main_terms = self.Ni_eos * delta ** self.di * tau ** self.ti
        ti_array = np.array(self.ti)
        extra_terms = ti_array * (np.array(self.ti) - 1.0)
        gaussian_terms = self.phi_i *(delta - self.D_i)**2.0 + self.beta_i * (tau - self.gamma_i)**2
        extra_gaussian_terms = (ti_array[self.m:self.n] - 2.0 * np.array(self.beta_i) * tau*(tau - np.array(self.gamma_i)))**2.0 - \
            ti_array[self.m:self.n] - 2.0*tau**2.0 *np.array(self.beta_i)
        d2_alphar_dtau2 = np.sum(extra_terms[:self.l]*main_terms[:self.l]) + \
            np.sum(main_terms[self.l:self.m]*np.exp(-(delta) ** self.pi[self.l:self.m])*extra_terms[self.l:self.m]) + \
            np.sum(main_terms[self.m:self.n]*np.exp(-gaussian_terms)*extra_gaussian_terms)
        return d2_alphar_dtau2

    def delta_tau_diff2_alphar_diff_delta_tau(self,rho,T):
        """delta * tau * d2(alpha_r)/(d(delta) d(tau))."""
        delta = rho / self.rho_c
        tau = self.Tc / T
        main_terms = self.Ni_eos * delta ** self.di * tau ** self.ti
        li = self.pi[self.l:self.m]
        extra_terms = (self.di[self.l:self.m] - li * (delta) ** li) * self.ti[self.l:self.m]
        gaussian_terms = self.phi_i *(delta - self.D_i)**2.0 + self.beta_i * (tau - self.gamma_i)**2.0
        extra_gaussian_terms = (self.di[self.m:self.n] - 2.0 * np.array(self.phi_i) * delta *(delta - np.array(self.D_i))) * \
            (self.ti[self.m:self.n] - 2.0 * np.array(self.beta_i) * tau*(tau - np.array(self.gamma_i)))
        d2_alphar_delta_tau = np.sum(np.array(self.di[:self.l]) * np.array(self.ti[:self.l]) * main_terms[:self.l]) + \
            np.sum(main_terms[self.l:self.m]*np.exp(-(delta) ** li)*extra_terms) + \
            np.sum(main_terms[self.m:self.n]*np.exp(-gaussian_terms)*extra_gaussian_terms)
        return d2_alphar_delta_tau

    def delta2_diff2_alphar_diffdelta2(self,rho,T):
        """delta^2 * d2(alpha_r)/d(delta)2."""
        delta = rho / self.rho_c
        tau = self.Tc / T
        main_terms = self.Ni_eos * delta ** self.di * tau ** self.ti
        li = self.pi[self.l:self.m]
        di_array = np.array(self.di)
        extra_terms = (di_array[self.l:self.m] - li * delta ** li) * \
            (di_array[self.l:self.m] - 1.0 - li * delta ** li) - \
            li**2.0 * delta ** li
        gaussian_terms = self.phi_i *(delta - self.D_i)**2.0 + self.beta_i * (tau - self.gamma_i)**2.0
        extra_gaussian_terms = (self.di[self.m:self.n] - 2.0 * self.phi_i * delta *(delta - self.D_i))**2.0 - \
            self.di[self.m:self.n] - 2.0*delta**2.0 *self.phi_i
        d2_alphar_ddelta2 = np.sum(self.di[:self.l] * (self.di[:self.l] - 1) * main_terms[:self.l]) + \
            np.sum(main_terms[self.l:self.m]*np.exp(-(delta) ** li)*extra_terms) + \
            np.sum(main_terms[self.m:self.n]*np.exp(-gaussian_terms)*extra_gaussian_terms)

        return d2_alphar_ddelta2


class NormalHydrogen(HydrogenData):
    """Normal hydrogen (75 % ortho): used for viscosity and, with para, for
    thermal conductivity.
    """
    def __init__(self):

        # ideal-gas (ak, bk) and residual (Ni, ti, di, pi, Gaussian) coefficients, Leachman et al.
        self.ak = np.array([-1.457985648,	1.888076782,	1.616,	-0.4117,	-0.792,	0.758,	1.217])
        self.bk = np.array([np.nan,	np.nan,	-16.02051591,	-22.6580178,	-60.00905114,	-74.94343038,	-206.9392065])

        self.Ni_eos = [-6.93643,	0.01,	2.11101,	4.52059,	0.732564,	-1.34086,	0.130985,	-0.777414,	0.351944,	-0.0211716,	0.0226312,	0.032187,	-0.0231752,	0.0557346]
        self.ti = np.array([0.6844,	1.0,	0.989,	0.489,	0.803,	1.1444,	1.409,	1.754,	1.311,	4.187,	5.646,	0.791,	7.249,	2.986])
        self.di = np.array([1,	4,	1,	1,	2,	2,	3,	1,	3,	2,	1,	3,	1,	1])
        self.pi = np.array([0,	0,	0,	0,	0,	0,	0,	1,	1,	np.nan, np.nan, np.nan, np.nan, np.nan])

        self.phi_i = np.array([1.685,	0.489,	0.103,	2.506,	1.607])
        self.beta_i = np.array([0.171,	0.2245,	0.1304,	0.2785,	0.3967])
        self.gamma_i = np.array([0.7164,	1.3444,	1.4517,	0.7204,	1.5445])
        self.D_i = np.array([1.506,	0.156,	1.736,	0.67,	1.662])

        # critical point: K, Pa, mol/m3
        self.Tc = 33.145
        self.Pc = 1.2964e6
        self.rho_c = 15.508e3


        # thermal conductivity coefficients (Assael et al.) and critical-enhancement cutoff
        self.tc_A1 = np.array([-3.40976E-01,	4.58820E+00,	-1.45080E+00,	3.26394E-01,	3.16939E-03,	1.90592E-04,	-1.13900E-06])
        self.tc_A2 = np.array([1.38497E+02,	-2.21878E+01,	4.57151E+00,	1.00000E+00])

        self.tc_B1 = np.array([3.63081E-02,	-2.07629E-02,	3.14810E-02,	-1.43097E-02,	1.74980E-03])
        self.tc_B2 = np.array([1.83370E-03,	-8.86716E-03,	1.58260E-02,	-1.06283E-02,	2.80673E-03])


        self.qd = (4e-10)**(-1.0)

        HydrogenData.__init__(self)


class OrthoHydrogen(HydrogenData):
    """Ortho-hydrogen: density, enthalpy and cp of the ortho fraction."""
    def __init__(self):

        # ideal-gas (ak, bk) and residual (Ni, ti, di, pi, Gaussian) coefficients, Leachman et al.
        self.ak = np.array([-1.4675442336, 1.8845068862, 2.54151, -2.3661, 1.00365, 1.22447])
        self.bk = np.array([np.nan, np.nan, -25.7676098736, -43.4677904877, -66.0445514750, -209.7531607465])

        self.Ni_eos = [-6.83148,		0.01,		2.11505,		4.38353,		0.211292,		-1.00939,		0.142086,	-0.87696,	0.804927,	-0.710775,	0.0639688,	0.0710858,	-0.087654,	0.647088]
        self.ti = np.array([0.7333,	1.0,	1.1372,	0.5136,	0.5638,	1.6248,	1.829,	2.404,	2.105,	4.1,	7.658,	1.259,	7.589,	3.946])
        self.di = np.array([1,	4,	1,	1,	2,	2,	3,	1,	3,	2,	1,	3,	1,	1])
        self.pi = np.array([0,	0,	0,	0,	0,	0,	0,	1,	1,	np.nan, np.nan, np.nan, np.nan, np.nan])

        self.phi_i = np.array([1.169,	0.894,	0.04,	2.072,	1.306])
        self.beta_i = np.array([0.4555,	0.4046,	0.0869,	0.4415,	0.5743])
        self.gamma_i = np.array([1.5444,	0.6627,	0.763,	0.6587,	1.4327])
        self.D_i = np.array([0.6366,	0.3876,	0.9437,	0.3976,	0.9626])

        # critical point: K, Pa, mol/m3
        self.Tc = 33.22
        self.Pc = 1.31065*1e6
        self.rho_c = 15.445e3


        HydrogenData.__init__(self)


class ParaHydrogen(HydrogenData):
    """Para-hydrogen: density, enthalpy, cp and thermal conductivity of the para fraction."""

    # Ortho -> para conversion enthalpy of the ideal gas at the reference temperature
    # (50 K), J/mol: Woolley, Scott & Brickwedde (1948), Table 4,
    # (H - E0)_para - (H - E0)_ortho = 248.581 - 587.041 cal/mol = -338.460 cal/mol
    # (x 4.184 J/cal). Aligns the para and ortho enthalpy scales.
    CONVERSION_ENTHALPY_REF = -1416.12
    def __init__(self):

        # ideal-gas (ak, bk) and residual (Ni, ti, di, pi, Gaussian) coefficients, Leachman et al.
        self.ak = np.array([-1.4485891134, 1.884521239, 	4.30256, 	13.0289, 	-47.7365, 	50.0013, 	-18.6261, 	0.993973, 	0.536078])
        self.bk = np.array([np.nan, np.nan, -15.14967515, 	-25.09259821, 	-29.47355638, 	-35.40591414, 	-40.72499848, 	-163.79258, 	-309.2173174])

        self.Ni_eos = [-7.33375, 	0.01, 	2.60375, 	4.66279, 	0.68239, 	-1.47078, 	0.135801, 	-1.05327, 	0.328239, 	-0.0577833, 	0.0449743, 	0.0703464, 	-0.0401766, 	0.11951]
        self.ti = np.array([0.6855, 	1.0, 	1.0, 	0.489, 	0.774, 	1.133, 	1.386, 	1.619, 	1.162, 	3.96, 	5.276, 	0.99, 	6.791, 	3.19])
        self.di = np.array([1, 	4, 	1, 	1, 	2, 	2, 	3, 	1, 	3, 	2, 	1, 	3, 	1, 	1])
        self.pi = np.array([0, 	0, 	0, 	0, 	0, 	0, 	0, 	1, 	1, np.nan, np.nan, np.nan, np.nan, np.nan])

        self.phi_i = np.array([1.7437, 	0.5516, 	0.0634, 	2.1341, 	1.777])
        self.beta_i = np.array([0.194, 	0.2019, 	0.0301, 	0.2383, 	0.3253])
        self.gamma_i = np.array([0.8048, 	1.5248, 	0.6648, 	0.6832, 	1.493])
        self.D_i = np.array([1.5487, 	0.1785, 	1.28, 	0.6319, 	1.7104])

        # critical point: K, Pa, mol/m3
        self.Tc = 32.938
        self.Pc = 1.2858*1e6
        self.rho_c = 15.538e3


        # thermal conductivity coefficients (Assael et al.) and critical-enhancement cutoff
        self.tc_A1 = np.array([-1.24500E+00, 	3.10212E+02, 	-3.31004E+02, 	2.46016E+02, 	-6.57810E+01, 	1.08260E+01, 	-5.19659E-01, 	1.43979E-02])
        self.tc_A2 = np.array([1.42304E+04, 	-1.93922E+04, 	1.58379E+04, 	-4.81812E+03, 	7.28639E+02, 	-3.57365E+01, 	1.00000E+00])

        self.tc_B1 = np.array([2.65975E-02, 	-1.33826E-03, 	1.30219E-02, 	-5.67678E-03, 	-9.23380E-05])
        self.tc_B2 = np.array([-1.21727E-03, 	3.66663E-03, 	3.88715E-03, 	-9.21055E-03, 	4.00723E-03])


        self.qd = (5e-10)**(-1.0)

        HydrogenData.__init__(self)

        # place para's enthalpy below ortho's by the tabulated conversion enthalpy at the
        # reference state; the equations of state supply the T and P dependence
        self.h_ref -= self.CONVERSION_ENTHALPY_REF


