"""Plate-fin heat exchanger: geometry, heat transfer and pressure drop.

PlateFinHex turns the configuration's 'reactor' block and the Topology into
channel counts, flow areas and heated areas, evaluates each stream's film
coefficient and pressure gradient (offset-strip fins: Manglik-Bergles;
catalyst-filled channels: Martin's wall coefficient with a Zehner-Bauer-
Schluender bed conductivity, and Ergun's pressure drop), and the heat
duty across every interface. bvp_model.bvp_function calls it at every node.
"""

import numpy as np
from functools import cached_property
from hydrogen_pfhx import heat_transfer_models
from hydrogen_pfhx.topology import PACKED_BED


class PlateFinHex(object):

    """Exchanger geometry and the per-stream transport results (h, f, k_bed,
    surface efficiencies) of the most recent evaluation.
    """
    def __init__(self, reactor_configuration, topology):
        self.topology = topology

        # per-stream results of the latest transport evaluation (filled at every node)
        self.h = {}
        self.f = {}
        self.k_bed = {}
        self.h_bare = {}
        self.eta_two = {}
        self.eta_one = {}

        # core geometry from the configuration (m)
        self.length = reactor_configuration['length']
        self.width = reactor_configuration['width']
        self.height = reactor_configuration['height']
        self.fin_thickness = reactor_configuration['fin_thickness']
        self.fin_pitch = reactor_configuration['fin_pitch']
        self.fin_height = reactor_configuration['fin_height']
        self.seration_length = reactor_configuration['seration_length']
        self.parting_sheet_thickness = reactor_configuration['parting_sheet_thickness']

    @cached_property
    def fin_spacing(self):
        """Free gap between neighbouring fins, s = pitch - thickness (m)."""
        return self.fin_pitch - self.fin_thickness

    @cached_property
    def hydraulic_perimeter(self):
        """Wetted perimeter of one channel: two fin flanks plus two plate faces (m)."""
        return 2 * (self.fin_height + self.fin_spacing)

    @cached_property
    def single_channel_area(self):
        """Free-flow cross-section of one channel (m2)."""
        return self.fin_spacing * self.fin_height

    @cached_property
    def hydraulic_diameter(self):
        """Plain-channel hydraulic diameter 4A/P (m): used for the catalyst-filled
        channels (Martin's D_t and the bed conduction term).
        """
        return 4 * self.single_channel_area / self.hydraulic_perimeter

    @cached_property
    def offset_strip_hydraulic_diameter(self):
        """Hydraulic diameter of Manglik & Bergles (1995), including the fin edges
        exposed at every serration: 4 s h l / (2 (s l + h l + t h) + t s). Their
        f and j were fitted with it, so it is used for Re, h and dP of fin streams.
        """
        s, h = self.fin_spacing, self.fin_height
        l, t = self.seration_length, self.fin_thickness
        return 4 * s * h * l / (2 * (s * l + h * l + t * h) + t * s)

    def _wall_temperature(self, label, stream, temperatures):
        """Estimated wall temperature of a stream: the mean of its own temperature
        and the contact-area-weighted mean of its neighbours. Only used to
        evaluate the fin metal's conductivity.
        """
        shares = self.topology.area_share[label]
        neighbours = self.topology.neighbours[label]
        # contact-area-weighted mean neighbour temperature
        weight = sum(shares[n] for n in neighbours)
        neighbour_mean = sum(shares[n] * temperatures[n] for n in neighbours) / weight
        return (stream.temperature + neighbour_mean) / 2

    @cached_property
    def layer_height(self):
        """Height of one layer: fin height + fin thickness + parting sheet (m)."""
        return self.fin_height + self.fin_thickness + self.parting_sheet_thickness

    @cached_property
    def number_layers(self):
        """Whole repeats of layer_order that fit in the stack height."""
        # round down to whole repeats so every stream keeps its share of layers
        layers_per_period = self.topology.n_positions
        return np.floor(self.height / self.layer_height / layers_per_period) * layers_per_period

    @cached_property
    def total_channels(self):
        """Channels over all layers: (width / pitch) per layer."""
        channels_per_layer = np.floor(self.width / self.fin_pitch)
        return channels_per_layer * self.number_layers

    def channels(self, label):
        """Channels belonging to one stream."""
        return self.topology.channel_fraction[label] * self.total_channels

    def total_side_area(self, label):
        """Total free-flow cross-section of one stream (m2)."""
        return self.single_channel_area * self.channels(label)

    @cached_property
    def fin_area_fraction(self):
        """Share of the channel perimeter formed by fins (the rest is plate)."""
        return 2 * self.fin_height / self.hydraulic_perimeter

    def interface_area_per_length(self, label_a, label_b):
        """CONTACT area per unit length between two streams (m2/m), from the
        topology's area_share; equal from both sides (check_area_consistency).
        """
        share = self.topology.area_share[label_a].get(label_b, 0.0)
        return share * self.hydraulic_perimeter * self.channels(label_a)

    def heated_area_per_length(self, label, neighbour):
        """(two-sided, one-sided) HEATED perimeter area per unit length (m2/m) of
        `label` facing `neighbour`; differs from the contact area for one-sided
        layers (see topology.Topology._build_geometry).
        """
        scale = self.hydraulic_perimeter * self.channels(label)
        return (self.topology.heated_share_two.get(label, {}).get(neighbour, 0.0) * scale,
                self.topology.heated_share_one.get(label, {}).get(neighbour, 0.0) * scale)

    def check_area_consistency(self, rtol=1e-9):
        """Raise if an interface's contact area differs depending on the side it
        is computed from (a topology inconsistency). Called once by model.model.
        """
        for a, b in self.topology.interfaces:
            area_ab = self.interface_area_per_length(a, b)
            area_ba = self.interface_area_per_length(b, a)
            if abs(area_ab - area_ba) > rtol * max(abs(area_ab), 1e-30):
                raise ValueError(
                    "interface %s<->%s area mismatch: %.6e from %s vs %.6e "
                    "from %s" % (a, b, area_ab, a, area_ba, b))

    def calculate_heat_transfer_duty(self, temperatures):
        """Heat flow per unit length (W/m) across every interface at one node,
        positive from the first to the second stream of the pair. Requires
        transport_behaviour to have been evaluated for all streams at this node.
        """
        duties = {}
        terms = {}
        for a, b in self.topology.interfaces:
            # wall between the two streams taken at their mean temperature
            wall_temperature = (temperatures[a] + temperatures[b]) / 2
            ua, resistance_terms = self.interface_conductance(a, b, wall_temperature)
            duties[(a, b)] = ua * (temperatures[a] - temperatures[b])
            terms[(a, b)] = resistance_terms
        return duties, terms

    def interface_conductance(self, label_a, label_b, wall_temperature):
        """UA per unit length (W/(m K)) of one interface:

            1/UA = 1/G_a + 1/G_b + (R_bed,a + R_bed,b + R_wall) / A_contact

        G = h_bare (eta_two A_two + eta_one A_one) is each side's film conductance
        over its own heated area; the parting sheet and the radial conduction
        through a catalyst bed are per contact area. Returns (UA, resistances).
        """
        solid_thermal_conductivity = np.mean(
            heat_transfer_models.aluminium_thermal_conductivity(wall_temperature))
        # series resistances per unit length: each side's film (own heated area) ...
        contact_area = self.interface_area_per_length(label_a, label_b)
        resistances = []
        per_contact = 0.0
        for label, other in ((label_a, label_b), (label_b, label_a)):
            area_two, area_one = self.heated_area_per_length(label, other)
            conductance = self.h_bare[label] * (
                self.eta_two[label] * area_two + self.eta_one[label] * area_one)
            resistances.append(1 / conductance)
            if self.topology.is_packed(label):
                # ... radial conduction through a catalyst bed, per contact area ...
                per_contact = per_contact + self.hydraulic_diameter / (8 * self.k_bed[label])
        # ... and conduction through the parting sheet
        per_contact = per_contact + self.parting_sheet_thickness / solid_thermal_conductivity
        resistances.append(per_contact / contact_area)
        return 1 / sum(resistances), resistances

    def overall_heat_transfer_coefficient(self, label_a, label_b, wall_temperature):
        """Overall U per unit CONTACT area (W/(m2 K)); reported in the results."""
        ua, terms = self.interface_conductance(label_a, label_b, wall_temperature)
        return ua / self.interface_area_per_length(label_a, label_b), terms

    def overall_surface_efficiency(self, h, k_fin):
        """Overall surface efficiency of a two-sided layer:
        eta_o = 1 - (A_fin/A) (1 - eta_fin), fins as two half-height fins.
        """
        eta_fin = self.calculate_fin_efficiency(h, k_fin)
        return 1 - self.fin_area_fraction * (1 - eta_fin)

    def overall_surface_efficiency_one_sided(self, h, k_fin):
        """Overall surface efficiency of a one-sided layer: fins heated from one
        wall only, conducting over their full height to an insulated far wall,
        which is counted as finned surface (it sits at fin-tip temperature).
        """
        eta_fin = self.calculate_fin_efficiency(h, k_fin, fin_length=self.fin_height)
        finned_fraction = (2 * self.fin_height + self.fin_spacing) / self.hydraulic_perimeter
        return 1 - finned_fraction * (1 - eta_fin)

    def calculate_fin_efficiency(self, h, k_fin, fin_length=None):
        """Straight-fin efficiency tanh(mL)/(mL), m = sqrt(2h / (k_fin t));
        L = half the fin height by default, the full height for one-sided layers.
        """
        # no fin correction for a vanishing film coefficient
        if h <= 0 or k_fin <= 0:
            return 1.0
        m = np.sqrt((2.0 * h) / (k_fin * self.fin_thickness))
        mb = m * (self.fin_height / 2.0 if fin_length is None else fin_length)
        return np.tanh(mb) / mb

    def transport_behaviour(self, label, stream, catalyst, temperatures):
        """Evaluate the film coefficient (and friction or bed conductivity) of one
        stream at the current node, by stream type.
        """
        if self.topology.is_packed(label):
            self._packed_bed_transport_behaviour(
                label, stream, catalyst, temperatures)
        else:
            self._offset_strip_transport_behaviour(
                label, stream, self._wall_temperature(label, stream, temperatures))

    def _packed_bed_transport_behaviour(self, label, stream, catalyst, temperatures):
        """Catalyst-filled channel: Martin's wall coefficient
            Nu_w = (1.3 + 5 Dp/Dt) k_bed/k_f + 0.19 Re_p^0.75 Pr^0.33,  Nu_w = h_w Dp / k_f
        with Re_p on the superficial velocity and particle diameter, and k_bed
        from effective_radial_conductivity. Stores h_bare, both surface
        efficiencies and the effective h.
        """
        # fin conductivity at the wall temperature, bed conductivity at the stream state
        wall_temperature = self._wall_temperature(label, stream, temperatures)
        k_fin = heat_transfer_models.aluminium_thermal_conductivity(wall_temperature)

        self.k_bed[label] = self.effective_radial_conductivity(
            stream, catalyst, self.hydraulic_diameter)
        Pr = stream.viscosity * stream.specific_heat_capacity / \
            stream.thermal_conductivity

        # particle Reynolds number on the superficial (empty-channel) velocity
        u_superficial = catalyst.void_fraction * stream.velocity
        Re = (stream.mass_density * u_superficial * catalyst.particle_diameter
              / stream.viscosity)
        stream.reynolds_number = Re

        Dp = catalyst.particle_diameter
        Dt = self.hydraulic_diameter
        # Martin's wall Nusselt number, defined with the particle diameter
        Nu_wo = (1.3 + 5 * Dp / Dt) * (self.k_bed[label] / stream.thermal_conductivity)
        Nu = Nu_wo + 0.19 * Re**0.75 * Pr**0.33

        h_w = Nu * stream.thermal_conductivity / Dp
        # store the bare h, both surface efficiencies and the effective (two-sided) h
        self.h_bare[label] = h_w
        self.eta_two[label] = self.overall_surface_efficiency(h_w, k_fin)
        self.eta_one[label] = self.overall_surface_efficiency_one_sided(h_w, k_fin)
        self.h[label] = self.eta_two[label] * h_w

    def _offset_strip_transport_behaviour(self, label, stream, wall_temperature):
        """Offset-strip fin channel: Manglik-Bergles j and f, Nu = j Re Pr^(1/3).
        Stores h_bare, both surface efficiencies, the effective h and f.
        """
        # Manglik-Bergles geometry ratios s/h, t/l, t/s
        alpha = self.fin_spacing / self.fin_height
        delta = self.fin_thickness / self.seration_length
        gamma = self.fin_thickness / self.fin_spacing
        (f, j) = heat_transfer_models.manglik_bergles_heat_transfer_model(
            stream.reynolds_number, alpha, delta, gamma)
        # film coefficient from j; fin efficiency at the wall temperature
        Nu = j * stream.reynolds_number * stream.prandtl_number**(1/3)
        h = Nu * stream.thermal_conductivity / self.offset_strip_hydraulic_diameter
        k_fin = heat_transfer_models.aluminium_thermal_conductivity(wall_temperature)
        self.h_bare[label] = h
        self.eta_two[label] = self.overall_surface_efficiency(h, k_fin)
        self.eta_one[label] = self.overall_surface_efficiency_one_sided(h, k_fin)
        h = h * self.eta_two[label]

        self.f[label] = f
        self.h[label] = h

    def pressure_drop(self, label, stream, catalyst):
        """Pressure gradient dP/dz (Pa/m) of one stream: the correlation gives the
        magnitude, the flow direction the sign (pressure falls along the flow).
        """
        if self.topology.is_packed(label):
            magnitude = -self._packed_bed_pressure_drop(stream, catalyst)
        else:
            magnitude = self._offset_strip_pressure_drop(label, stream)
        return -self.topology.flow_sign(label) * magnitude

    def _packed_bed_pressure_drop(self, stream, catalyst):
        """Frictional gradient (Pa/m, negative) of a catalyst bed (Ergun)."""
        voidage = catalyst.void_fraction
        # superficial velocity from the interstitial stream velocity
        vs = voidage * stream.velocity
        Dp = catalyst.particle_diameter
        rho = stream.mass_density
        # Ergun: viscous plus inertial term
        return -(150*stream.viscosity * (1-voidage)**2 * vs / (Dp**2 * voidage**3)
                 + 1.75 * rho * (1 - voidage) * vs**2 / (Dp * voidage**3))

    def _offset_strip_pressure_drop(self, label, stream):
        """Frictional gradient magnitude (Pa/m) of a fin channel: 2 f rho u^2 / D_h (Fanning)."""
        return 2 * self.f[label] * stream.mass_density * \
            stream.velocity**2 / self.offset_strip_hydraulic_diameter

    def effective_radial_conductivity(self, stream, catalyst, d_t):
        """Effective conductivity of the catalyst bed (W/(m K)): stagnant part from
        Zehner-Bauer-Schluender with radiation (VDI Heat Atlas, D6.3; continuum
        gas, flattening coefficient for spheres), plus radial dispersion from the
        flow. Used in Martin's wall coefficient and the bed resistance.
        """
        k_f = stream.thermal_conductivity
        k_s = catalyst.calculate_thermal_conductivity(stream.temperature)
        epsilon = catalyst.void_fraction
        dp = catalyst.particle_diameter

        # radiation: 4 sigma T^3 / (2/zeta - 1) with emissivity zeta (negligible at these temperatures)
        zeta = 0.9
        alpha_rs = 0.2268 * (zeta/(2-zeta))*(stream.temperature/100)**3
        k_rad = alpha_rs * dp / k_f
        # ZBS: conductivity ratio, flattening coefficient phi, shape factor B, and core term k_c
        k_p = k_s / k_f
        phi = 0.0077
        B = 1.25 * ((1 - epsilon) / epsilon)**(10/9)
        N = 1 + (k_rad - B) / k_p
        k_c = 2 / N * (B * (k_p + k_rad - 1) / (N**2 * k_p) * np.log((k_p + k_rad) / B)
                       + (B + 1) / (2 * B) * (k_rad - B) - (B - 1) / N)
        # stagnant bed conductivity: gas-filled pores plus the particle-contact core
        k_0 = k_f * ((1 - (1 - epsilon)**0.5) * (1 + epsilon * k_rad)
                     + (1 - epsilon)**0.5 * (phi * k_p + (1 - phi) * k_c))

        # radial dispersion by the flow, on the superficial velocity (eps * interstitial)
        Pe = (1 + 46*(dp/d_t)**2)/0.14
        D_er = epsilon * stream.velocity * dp / (epsilon * Pe)
        k_t = epsilon * stream.mass_density * stream.specific_heat_capacity * D_er

        return k_0 + k_t
