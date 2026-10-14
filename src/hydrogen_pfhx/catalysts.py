"""Catalyst bed description used by the packed-bed (catalysed) streams:
geometry for the flow and heat-transfer correlations, particle conductivity
for the bed conductivity.
"""


# Ionex-type hydrous ferric oxide particles, W/(m K) (Li et al., 2023, Table 7)
DEFAULT_THERMAL_CONDUCTIVITY = 0.58


class Catalyst(object):
    """Catalyst-filled channel properties, built from the configuration's
    'catalyst' block and read by hexs.PlateFinHex and bvp_model.
    """
    def __init__(self, catalyst_configuration):
        # bed structure: solid volume fraction, particle diameter (m) and porosity
        self.solid_fraction = catalyst_configuration['solid_fraction']
        self.particle_diameter = catalyst_configuration['diameter']
        self.void_fraction = 1 - catalyst_configuration['solid_fraction']
        # particle conductivity; the configuration may override the default
        self.thermal_conductivity = catalyst_configuration.get(
            'thermal_conductivity', DEFAULT_THERMAL_CONDUCTIVITY)

    def calculate_thermal_conductivity(self, temperature):
        """Particle thermal conductivity (W/(m K)), used in the bed conductivity
        (hexs.effective_radial_conductivity); taken as temperature-independent.
        """
        return self.thermal_conductivity
