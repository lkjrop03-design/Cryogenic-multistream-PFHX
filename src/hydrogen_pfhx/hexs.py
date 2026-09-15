import numpy as np
from functools import cached_property
from hydrogen_pfhx import heat_transfer_models, pressure_models
from hydrogen_pfhx.topology import PACKED_BED


class PlateFinHex(object):
    """Generic multi-stream plate-fin heat exchanger.

    All structure comes from the `Topology` object (see topology.py):
    channel fractions, which streams exchange heat, how each channel's
    wetted perimeter is split between its neighbours, and the state
    layout. Nothing about stream count or stacking order is hard-coded
    here.

    Two stream types are dispatched on:
      PACKED_BED - catalyst-filled: Ergun friction factor, effective
        radial (bed) conductivity, wall-temperature-corrected Nusselt
        via Gnielinski with a laminar/transition/turbulent branch.
      PLAIN_FIN  - unpacked: Manglik-Bergles f and j, no wall-temperature
        dependence.

    Per-stream transport state is stored in dicts keyed by stream label
    (self.h[label], self.f[label], self.k_bed[label]) rather than the
    h_hot/h_cold_1/... attributes the hand-written topologies used.
    """

    def __init__(self, reactor_configuration, topology):
        self.topology = topology

        # per-stream transport state, filled by *_transport_behaviour
        self.h = {}          # effective (fin-corrected) film coefficient
        self.f = {}          # friction factor
        self.k_bed = {}      # effective radial bed conductivity (packed only)

        # input
        self.length = reactor_configuration['length']
        self.width = reactor_configuration['width']
        self.height = reactor_configuration['height']
        self.fin_thickness = reactor_configuration['fin_thickness']
        self.fin_pitch = reactor_configuration['fin_pitch']
        self.fin_height = reactor_configuration['fin_height']
        self.seration_length = reactor_configuration['seration_length']
        self.parting_sheet_thickness = reactor_configuration['parting_sheet_thickness']

    # ------------------------------------------------------------------
    # geometry (unchanged from the hand-written versions)
    # ------------------------------------------------------------------
    @cached_property
    def fin_spacing(self):
        return self.fin_pitch - self.fin_thickness

    @cached_property
    def hydraulic_perimeter(self):
        return 2 * (self.fin_height + self.fin_spacing)

    @cached_property
    def single_channel_area(self):
        return self.fin_spacing * self.fin_height

    @cached_property
    def hydraulic_diameter(self):
        return 4 * self.single_channel_area / self.hydraulic_perimeter

    @cached_property
    def characteristic_length(self):
        return self.hydraulic_diameter

    @cached_property
    def layer_height(self):
        return self.fin_height + self.fin_thickness + self.parting_sheet_thickness

    @cached_property
    def number_layers(self):
        """Rounded DOWN to a whole number of periods, so the stack always
        terminates on a complete repeating unit - the period length now
        comes from the topology rather than being hard-coded to 4."""
        layers_per_period = self.topology.n_positions
        return np.floor(self.height / self.layer_height / layers_per_period) * layers_per_period

    @cached_property
    def total_channels(self):
        channels_per_layer = np.floor(self.width / self.fin_pitch)
        return channels_per_layer * self.number_layers

    def channels(self, label):
        """Number of channels belonging to `label`."""
        return self.topology.channel_fraction[label] * self.total_channels

    def total_side_area(self, label):
        """Total flow cross-sectional area for `label`."""
        return self.single_channel_area * self.channels(label)

    @cached_property
    def total_channel_area(self):
        return self.single_channel_area * self.total_channels

    @cached_property
    def fin_area_fraction(self):
        return 2 * self.fin_height / self.hydraulic_perimeter

    def interface_area_per_length(self, label_a, label_b):
        """Heat transfer area per unit reactor length (m2/m) for the
        interface between two streams.

        Computed from side `label_a`: (that stream's share of its own
        perimeter facing label_b) x perimeter x (its channel count).

        This is symmetric - computing from either side gives the same
        area - which is a genuine invariant of the topology derivation,
        not an assumption: see `check_area_consistency`.
        """
        share = self.topology.area_share[label_a].get(label_b, 0.0)
        return share * self.hydraulic_perimeter * self.channels(label_a)

    def check_area_consistency(self, rtol=1e-9):
        """Verifies each interface's area is the same computed from
        either side. Cheap, and catches any future topology-derivation
        mistake immediately rather than as a silent energy imbalance."""
        for a, b in self.topology.interfaces:
            area_ab = self.interface_area_per_length(a, b)
            area_ba = self.interface_area_per_length(b, a)
            if abs(area_ab - area_ba) > rtol * max(abs(area_ab), 1e-30):
                raise ValueError(
                    "interface %s<->%s area mismatch: %.6e from %s vs %.6e "
                    "from %s" % (a, b, area_ab, a, area_ba, b))

    # ------------------------------------------------------------------
    # heat transfer
    # ------------------------------------------------------------------
    def calculate_heat_transfer_duty(self, temperatures):
        """`temperatures`: {label: T}. Returns (duties, terms) where
        duties[(a, b)] is the linear heat duty (W/m) flowing from a to b
        for each interface in topology.interfaces, and terms[(a, b)] is
        that interface's list of resistance terms (for diagnostics).

        Sign convention: duties[(a, b)] > 0 means heat flows a -> b, so
        it is a LOSS for a and a GAIN for b. The caller applies each
        stream's own flow-direction sign.
        """
        duties = {}
        terms = {}
        for a, b in self.topology.interfaces:
            wall_temperature = (temperatures[a] + temperatures[b]) / 2
            u, resistance_terms = self.overall_heat_transfer_coefficient(
                a, b, wall_temperature)
            area = self.interface_area_per_length(a, b)
            duties[(a, b)] = u * area * (temperatures[a] - temperatures[b])
            terms[(a, b)] = resistance_terms
        return duties, terms

    def overall_heat_transfer_coefficient(self, label_a, label_b, wall_temperature):
        """Series resistance network for one interface, built ADDITIVELY:
          - one film-coefficient term per side
          - one parting-sheet conduction term
          - one packed-bed radial conduction term PER PACKED SIDE

        The last point is the generalisation: the hand-written versions
        had a fixed 4-term (packed-vs-plain) or 3-term (plain-vs-plain)
        form. Building it additively also covers a packed<->packed
        interface (two bed terms), which is reachable in a topology like
        [h1, c1, c2, h2] where two packed layers end up adjacent.
        """
        solid_thermal_conductivity = np.mean(
            heat_transfer_models.aluminium_thermal_conductivity(wall_temperature))

        terms = []
        for label in (label_a, label_b):
            if self.topology.is_packed(label):
                terms.append(self.hydraulic_diameter / (8 * self.k_bed[label]))
            terms.append(1 / self.h[label])
        terms.append(self.parting_sheet_thickness / solid_thermal_conductivity)

        u = 1 / np.sum(terms)
        return u, terms

    def calculate_fin_efficiency(self, h, k_fin):
        """Efficiency of a straight fin."""
        if h <= 0 or k_fin <= 0:
            return 1.0
        m = np.sqrt((2.0 * h) / (k_fin * self.fin_thickness))
        # b = HALF the fin height, per Shah & Sekulic's standard plate-fin
        # treatment: the two parting sheets bounding a fin channel sit at
        # nearly the same temperature, so the fin's midplane carries zero
        # heat flux by symmetry - exactly like a true adiabatic tip - and
        # each fin is modelled as two half-height fins back-to-back.
        # Using the full height systematically UNDERESTIMATES fin
        # efficiency and therefore every effective film coefficient and
        # overall U.
        mb = m * (self.fin_height / 2.0)
        return np.tanh(mb) / mb

    # ------------------------------------------------------------------
    # transport behaviour - dispatched on stream type
    # ------------------------------------------------------------------
    def transport_behaviour(self, label, stream, catalyst, temperatures):
        """Computes and stores f, h (and k_bed for packed streams) for
        one stream, dispatching on its type."""
        if self.topology.is_packed(label):
            self._packed_bed_transport_behaviour(
                label, stream, catalyst, temperatures)
        else:
            self._plain_fin_transport_behaviour(label, stream)

    def _packed_bed_transport_behaviour(self, label, stream, catalyst, temperatures):
        """Martin et al.'s packed-bed correlation (constant-q_w variant),
        replacing the generic duct-flow (Shah & Sekulic / Gnielinski)
        blend previously used here.

            Nu_w  = Nu_wo + 0.19 * Re_p^0.75 * Pr^0.33
            Nu_wo = (1.3 + 5*Dp/Dt) * (k_bed/k_f)
            h_w   = Nu_w * k_f / Dt

        Dt is taken as the channel hydraulic diameter (the same
        tube-diameter-to-hydraulic-diameter adaptation used throughout
        this model for non-circular ducts). k_bed is effective_radial_
        conductivity - the SAME ZBS effective conductivity already used
        for the bed's own conduction resistance (see
        overall_heat_transfer_coefficient's packed-bed term, which is
        already Martin's h_bed formula for constant heat flux,
        h_bed = 8*k_bed/Dt, in resistance form). h_w's own reference
        conductivity is k_f (the fluid's own conductivity), NOT k_bed -
        this is a genuine difference from the previous h_hot
        (Nu*k_bed/D_h) and easy to get backwards.

        No wall-temperature-corrected Prandtl number is needed here
        (Martin's correlation doesn't use one), so wall_temperature is
        not computed in this function at all - it is still needed
        elsewhere (the parting-sheet conduction term), just not here.
        """
        self.k_bed[label] = self.effective_radial_conductivity(
            stream, catalyst, self.hydraulic_diameter)
        Pr = stream.viscosity * stream.specific_heat_capacity / \
            stream.thermal_conductivity

        stream.calculate_reynolds_number(catalyst.particle_diameter)
        Re = stream.reynolds_number

        self.f[label] = pressure_models.ergun_equation(catalyst.void_fraction, Re)

        if Re < 0:
            print('Cannot have negative Reynolds')

        Dp = catalyst.particle_diameter
        Dt = self.hydraulic_diameter
        Nu_wo = (1.3 + 5 * Dp / Dt) * (self.k_bed[label] / stream.thermal_conductivity)
        Nu = Nu_wo + 0.19 * Re**0.75 * Pr**0.33

        h_w = Nu * stream.thermal_conductivity / Dt
        self.h[label] = self.calculate_fin_efficiency(
            h_w, stream.thermal_conductivity) * h_w

    def _plain_fin_transport_behaviour(self, label, stream):
        alpha = self.fin_spacing / self.fin_height
        delta = self.fin_thickness / self.seration_length
        gamma = self.fin_thickness / self.fin_spacing
        (f, j) = heat_transfer_models.manglik_bergles_heat_transfer_model(
            stream.reynolds_number, alpha, delta, gamma)
        Nu = j * stream.reynolds_number * stream.prandtl_number**(1/3)
        h = Nu * stream.thermal_conductivity / self.hydraulic_diameter
        h = h * self.calculate_fin_efficiency(h, stream.thermal_conductivity)

        self.f[label] = f
        self.h[label] = h

    # ------------------------------------------------------------------
    # pressure drop - dispatched on stream type
    # ------------------------------------------------------------------
    def pressure_drop(self, label, stream, catalyst):
        if self.topology.is_packed(label):
            return self._packed_bed_pressure_drop(stream, catalyst)
        return self._plain_fin_pressure_drop(label, stream)

    def _packed_bed_pressure_drop(self, stream, catalyst):
        voidage = catalyst.void_fraction
        vs = voidage * stream.velocity
        Dp = catalyst.particle_diameter
        rho = stream.mass_density
        return -(150*stream.viscosity * (1-voidage)**2 * vs / (Dp**2 * voidage**3)
                 + 1.75 * rho * (1 - voidage) * vs**2 / (Dp * voidage**3))

    def _plain_fin_pressure_drop(self, label, stream):
        return 2 * self.f[label] * stream.mass_density * \
            stream.velocity**2 / self.hydraulic_diameter

    # ------------------------------------------------------------------
    def effective_radial_conductivity(self, stream, catalyst, d_t):
        k_f = stream.thermal_conductivity
        k_s = catalyst.calculate_thermal_conductivity(stream.temperature)
        epsilon = catalyst.void_fraction
        dp = catalyst.particle_diameter

        zeta = 0.9
        alpha_rs = 2e-5 * (zeta/(2-zeta))*(stream.temperature/100)**3
        B = 1.25 * ((1 - epsilon) / epsilon)**(10/9)

        k_r = k_f/k_s
        dim_alpha = alpha_rs * dp / k_f

        omega = (1+(dim_alpha - 1)*k_r)/(1+(dim_alpha - B)*k_r) * np.log((1 + alpha_rs*dp/k_s) /
                                                                         (B*k_r)) - (B-1) / (1 + (dim_alpha - B)*k_r) + (B+1) / (2*B) * (dim_alpha - B)
        k_0 = k_f * ((1 - (1-epsilon))**0.5 * (1 + epsilon * dim_alpha) +
                     2*omega * (1-epsilon)**0.5/(1+(dim_alpha - B)*k_r))

        Pe = (1 + 46*(dp/d_t)**2)/0.14
        D_er = stream.velocity * dp / (epsilon * Pe)
        k_t = epsilon * stream.mass_density * stream.specific_heat_capacity * D_er

        return k_0 + k_t
