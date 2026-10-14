"""Fluid streams: state and thermophysical properties of every stream in the
exchanger. FluidStream wraps a CoolProp equation of state for any pure
fluid (helium, nitrogen, ...); Hydrogen builds its properties from separate
ortho and para equations of state (hydrogen_thermodynamics) mixed at the
stream's para fraction. bvp_model sets each stream's state at every node and
reads the properties for the transport, kinetics and energy balance.
"""

import numpy as np
import CoolProp.CoolProp as CP
from hydrogen_pfhx import ortho_para_dynamics, hydrogen_thermodynamics


class FluidStream(object):

    """One stream of a pure CoolProp fluid: inlet flow, current state and the
    properties derived from it.
    """
    def __init__(self, fluid, mass_flow_rate):
        self.fluid = fluid
        self.mass_flow_rate = mass_flow_rate

        # CoolProp equation of state and the fluid's lower limit (triple point)
        self.abstract_state = CP.AbstractState('HEOS', fluid)
        self.triple_point_K = self.abstract_state.Ttriple()

        # current state and the properties derived from it (set at every node)
        self.temperature = None
        self.pressure = None

        self.molecular_mass = None
        self.molar_density = None
        self.mass_density = None
        self.viscosity = None
        self.specific_heat_capacity = None
        self.thermal_conductivity = None
        self.prandtl_number = None
        self.enthalpy = None
        self.entropy = None
        self.molar_flow_rate = None

        self.velocity = None
        self.reynolds_number = None
        self.critical_temperature = None
        self.critical_pressure = None

    def update_conditions(self, temperature, pressure):
        """Set temperature (K) and pressure (Pa) of the stream's current state."""
        self.temperature = temperature
        self.pressure = pressure
        self.abstract_state.update(CP.PT_INPUTS, pressure, temperature)

    def set_properties(self):
        """Evaluate all properties at the current state (called after
        update_conditions, and update_composition for hydrogen).
        """
        self.molecular_mass = self._calculate_molecular_mass()
        self.molar_density = self._calculate_molar_density()
        self.mass_density = self.molar_density * self.molecular_mass
        self.viscosity = self._calculate_viscosity()
        self.specific_heat_capacity = self._calculate_specific_heat_capacity()
        self.thermal_conductivity = self._calculate_thermal_conductivity()
        # derived groups: Prandtl number and molar flow
        self.prandtl_number = self.viscosity * \
            self.specific_heat_capacity/self.thermal_conductivity
        self.enthalpy = self._calculate_enthalpy()
        self.entropy = self._calculate_entropy()
        self.molar_flow_rate = self.mass_flow_rate / self.molecular_mass
        self.critical_temperature = self._calculate_critical_temperature()
        self.critical_pressure = self._calculate_critical_pressure()

    def calculate_velocity(self, cross_sectional_area):
        """Mean velocity (m/s) through the given free-flow area (m2)."""
        self.velocity = self.mass_flow_rate / \
            (self.mass_density * cross_sectional_area)


    def calculate_reynolds_number(self, characteristic_length):
        """Reynolds number on the given length scale (m)."""
        self.reynolds_number = self.mass_density * \
            self.velocity * characteristic_length / self.viscosity

    # safety margin above the solid-phase limit (K)
    TEMPERATURE_FLOOR_MARGIN_K = 1.0

    def temperature_floor(self, pressure_pa):
        """Lowest temperature (K) the solver may pass to this fluid at pressure P
        (Pa): the triple point, or the melting temperature at P if higher, plus a
        margin. Used by bvp_model.update_parameters and model._check_inlet_limits.
        """
        T_limit = self.triple_point_K
        try:
            # melting temperature at this pressure; absent below the melting line (e.g. helium < 2.5 MPa)
            T_melt = self.abstract_state.melting_line(CP.iT, CP.iP, pressure_pa)
            T_limit = max(T_limit, T_melt)
        except ValueError:
            pass
        return T_limit + self.TEMPERATURE_FLOOR_MARGIN_K


    def _calculate_molar_density(self):
        rho = self.abstract_state.rhomolar()

        return rho

    def _calculate_viscosity(self):

        mu = self.abstract_state.viscosity()
        mu = self._bridge_helium_viscosity_seam(mu)

        return mu

    # location and half width (K) of the bridged helium viscosity step
    _HELIUM_SEAM_K = 100.0
    _HELIUM_SEAM_HALF_WIDTH_K = 2.0

    def _bridge_helium_viscosity_seam(self, mu):
        """Helium only: CoolProp's viscosity correlation steps by ~2 % at 100 K.
        Between 98 and 102 K it is replaced by a cubic Hermite bridge matching
        value and slope of the correlation on both sides, so the solver sees a
        smooth property.
        """
        T = self.temperature
        dT = self._HELIUM_SEAM_HALF_WIDTH_K
        if str(self.fluid).lower() != 'helium' or abs(T - self._HELIUM_SEAM_K) >= dT:
            return mu

        # separate CoolProp state so the stream's own state is not disturbed
        if not hasattr(self, '_seam_state'):
            self._seam_state = CP.AbstractState('HEOS', self.fluid)
        state = self._seam_state

        def mu_at(t):
            state.update(CP.PT_INPUTS, self.pressure, t)
            return state.viscosity()

        eps = 1e-3
        T0, T1 = self._HELIUM_SEAM_K - dT, self._HELIUM_SEAM_K + dT
        m0, m1 = mu_at(T0), mu_at(T1)
        # one-sided slopes of the correlation just outside the bridge
        d0 = (m0 - mu_at(T0 - eps)) / eps
        d1 = (mu_at(T1 + eps) - m1) / eps
        h = T1 - T0
        # cubic Hermite interpolation between (T0, m0, d0) and (T1, m1, d1)
        s = (T - T0) / h
        return ((2*s**3 - 3*s**2 + 1) * m0 + (s**3 - 2*s**2 + s) * h * d0
                + (-2*s**3 + 3*s**2) * m1 + (s**3 - s**2) * h * d1)

    def _calculate_specific_heat_capacity(self):
        cp = self.abstract_state.cpmass()

        return cp

    def _calculate_thermal_conductivity(self):
        k = self.abstract_state.conductivity()

        return k

    def _calculate_enthalpy(self):
        h = self.abstract_state.hmass()

        return h

    def _calculate_entropy(self):
        s = self.abstract_state.smass()
        return s

    def _calculate_molecular_mass(self):
        m = self.abstract_state.molar_mass()
        return m

    def _calculate_critical_temperature(self):
        Tc = self.abstract_state.T_critical()
        return Tc

    def _calculate_critical_pressure(self):
        Pc = self.abstract_state.p_critical()
        return Pc


class Hydrogen(FluidStream):
    """Hydrogen stream with variable ortho/para composition: densities from the
    ortho, para and normal equations of state, mixed at para fraction xp.
    """
    def __init__(self, mass_flow_rate):
        FluidStream.__init__(self, 'Hydrogen', mass_flow_rate)
        # separate equations of state for the two isomers, and normal hydrogen for transport properties
        self.ortho = hydrogen_thermodynamics.OrthoHydrogen()
        self.para = hydrogen_thermodynamics.ParaHydrogen()
        self.normal = hydrogen_thermodynamics.NormalHydrogen()

    def set_properties(self):
        """Solve the ortho, para and normal densities at the current T and P
        (CoolProp's normal-hydrogen density as starting guess), then evaluate the
        mixture properties.
        """
        rho_guess = self.abstract_state.rhomolar()
        self.ortho_density = self.ortho.calculate_density(
            self.pressure, self.temperature, rho_guess)
        self.para_density = self.para.calculate_density(
            self.pressure, self.temperature, rho_guess)
        self.normal_density = self.normal.calculate_density(
            self.pressure, self.temperature, rho_guess)

        FluidStream.set_properties(self)

    def update_composition(self, xp):
        """Set the para mole fraction (equal to the mass fraction for the isomers)."""
        self.xp = xp

    def get_xp(self):
        xp = self.xp
        return xp

    def get_xo(self):
        xo = 1.0 - self.xp
        return xo

    def get_xp_equil(self):
        """Equilibrium para fraction at the stream temperature."""
        (xpe, _) = ortho_para_dynamics.para_ortho_equilibrium(self.temperature)
        return xpe

    def get_xo_equil(self):
        xpo = 1.0 - self.get_xp_equil()
        return xpo

    def get_heat_of_conversion(self):
        """Ortho -> para conversion enthalpy at the current state (J/mol, negative
        = exothermic), the heat source of the catalysed stream's energy balance.
        """
        ortho_enthalpy = self.ortho.enthalpy(
            self.ortho_density, self.temperature)
        para_enthalpy = self.para.enthalpy(self.para_density, self.temperature)
        delta_h = para_enthalpy - ortho_enthalpy
        return delta_h


    def _calculate_molar_density(self):
        """Mixture molar density (mol/m3) by ideal mixing of the isomers' molar volumes."""
        rho = 1.0 / ((1 - self.xp) / self.ortho_density
                     + self.xp / self.para_density)
        return rho

    def _calculate_thermal_conductivity(self):
        """Conductivity interpolated between normal (xp = 0.25) and para (xp = 1)
        hydrogen; the critical enhancement uses CoolProp's background viscosity.
        """
        viscosity_contributions = self.abstract_state.viscosity_contributions()
        mu_background = viscosity_contributions['residual'] - \
            viscosity_contributions['critical']

        ph_k, _ = self.para.thermal_conductivity(
            self.para_density, self.temperature, mu_background)

        nh_k, _ = self.normal.thermal_conductivity(
            self.normal_density, self.temperature, mu_background)

        k = nh_k + (ph_k - nh_k) * (0.75 - self.get_xo())/0.75
        return k

    def _calculate_specific_heat_capacity(self):
        """Frozen-composition cp (J/(kg K)): mole-fraction average of ortho and
        para. The conversion heat is added separately in the energy balance.
        """
        ph_cp = self.para.specific_heat_constant_pressure(
            self.para_density, self.temperature)

        oh_cp = self.ortho.specific_heat_constant_pressure(
            self.ortho_density, self.temperature)

        cp_molar = (1-self.xp)*oh_cp + self.xp*ph_cp
        cp = cp_molar / self.molecular_mass
        return cp

    def _calculate_viscosity(self):
        """Viscosity of normal hydrogen (the isomers' viscosities are practically equal)."""
        v = self.normal.viscosity(
            self.normal_density, self.temperature, self.molecular_mass)
        return v

    def _calculate_enthalpy(self):
        """Mixture enthalpy (J/kg): mole-fraction average of the ortho and para
        enthalpies, including their conversion offset.
        """
        ph_h = self.para.enthalpy(self.para_density, self.temperature)

        oh_h = self.ortho.enthalpy(self.ortho_density, self.temperature)

        h_molar = (1-self.xp)*oh_h + self.xp*ph_h
        h = h_molar / self.molecular_mass
        return h

    def _calculate_entropy(self):
        """Mole-fraction average of ortho and para entropy (J/(kg K)); output only
        (no mixing term or ortho/para offset).
        """
        ph_s = self.para.entropy(self.para_density, self.temperature)

        oh_s = self.ortho.entropy(self.ortho_density, self.temperature)

        s_molar = (1-self.xp)*oh_s + self.xp*ph_s
        s = s_molar / self.molecular_mass
        return s

    def _calculate_critical_temperature(self):
        """Composition-weighted critical temperature (used by the property cache variants)."""
        Tc = self.get_xo()*self.ortho.Tc + self.get_xp()*self.para.Tc
        return Tc

    def _calculate_critical_pressure(self):
        """Composition-weighted critical pressure."""
        Pc = self.get_xo()*self.ortho.Pc + self.get_xp()*self.para.Pc
        return Pc
