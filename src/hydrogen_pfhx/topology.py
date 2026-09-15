"""
Topology / state-vector layer.

Derives EVERYTHING geometric and structural from one ordered list: the
physical stacking order of one repeating period of layers. The list is
treated as CYCLIC (the last layer's outer face abuts the first layer of
the next period), so each position's two thermal neighbours are simply
its list-neighbours.

This replaces the hand-written, per-topology `IDX_*` constants,
hard-coded channel fractions, hard-coded 50/50 perimeter splits and
hand-derived interface lists that every previous topology needed.

Verified to reproduce all three hand-built topologies exactly:
  ['c1','h','c2','h']    -> c1-h-c2-h    (hot is the "hub", 50/25/25)
  ['c','h1','c','h2']    -> c-h1-c-h2    (cold is the "hub", 50/25/25)
  ['c1','h1','c2','h2']  -> c1-h1-c2-h2  (no hub, 25/25/25/25, 4 interfaces)

Notes on the cyclic assumption
------------------------------
Rotating the list gives the same exchanger (it is a repeating stack), so
a topology has no canonical "first" layer. With exactly one packed-bed
layer you could always rotate it to position 0, but that normalisation
does NOT collapse the space once there are two or more packed layers -
the spacing between them is real information that rotation cannot
remove (e.g. ['h1','c1','h2','c2'] and ['h1','c1','c2','h2'] both start
with a packed layer and are genuinely different exchangers). This module
therefore makes no assumption about where packed layers sit.

The cyclic wrap is an idealisation: a real finite stack has two
outermost faces that abut the shell rather than another layer. This is
the same assumption every hand-built topology here already made (each
rounded the layer count down to a whole number of periods); it is
accurate when the stack is many periods tall, which it is.
"""

from collections import Counter, defaultdict


PACKED_BED = 'packed_bed'
PLAIN_FIN = 'plain_fin'

CO_CURRENT = 'co_current'
COUNTER_CURRENT = 'counter_current'

_VALID_TYPES = (PACKED_BED, PLAIN_FIN)
_VALID_DIRECTIONS = (CO_CURRENT, COUNTER_CURRENT)


class Topology(object):
    """Structural description of a plate-fin stack, derived from the
    ordered layer sequence.

    Parameters
    ----------
    layer_order : list of str
        One period of the physical stacking order, e.g.
        ['coolant_1', 'hot', 'coolant_2', 'hot']. Labels may repeat -
        a repeated label means that same stream occupies several
        layers per period.
    streams : dict
        {label: {'type': PACKED_BED|PLAIN_FIN,
                 'direction': CO_CURRENT|COUNTER_CURRENT, ...}}
        Extra keys (mass_flow_rate, inlet conditions, ...) are ignored
        here and left for the model layer to consume.
    """

    def __init__(self, layer_order, streams):
        self._validate(layer_order, streams)
        self.layer_order = list(layer_order)
        self.streams = streams

        # Stable, deterministic stream ordering: order of first
        # appearance in layer_order. Everything downstream (state vector
        # layout, interface ordering) keys off this, so the same config
        # always produces an identically-laid-out state vector.
        seen = []
        for label in self.layer_order:
            if label not in seen:
                seen.append(label)
        self.labels = seen

        self._counts = Counter(self.layer_order)
        self.n_positions = len(self.layer_order)

        self._build_geometry()
        self._build_state_map()

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate(layer_order, streams):
        if len(layer_order) < 2:
            raise ValueError(
                "layer_order needs at least 2 layers; got %r" % (layer_order,))

        missing = set(layer_order) - set(streams)
        if missing:
            raise ValueError(
                "layer_order references stream(s) with no definition in "
                "`streams`: %s" % sorted(missing))

        unused = set(streams) - set(layer_order)
        if unused:
            raise ValueError(
                "`streams` defines stream(s) that never appear in "
                "layer_order: %s" % sorted(unused))

        for label, spec in streams.items():
            stype = spec.get('type')
            if stype not in _VALID_TYPES:
                raise ValueError(
                    "stream %r has type %r; expected one of %s"
                    % (label, stype, list(_VALID_TYPES)))
            direction = spec.get('direction')
            if direction not in _VALID_DIRECTIONS:
                raise ValueError(
                    "stream %r has direction %r; expected one of %s"
                    % (label, direction, list(_VALID_DIRECTIONS)))

        # A stack where every layer is the same stream has no interfaces
        # and nothing to solve.
        if len(set(layer_order)) < 2:
            raise ValueError(
                "layer_order must contain at least 2 distinct streams; got %r"
                % (layer_order,))

    # ------------------------------------------------------------------
    # geometry derived from the cyclic order
    # ------------------------------------------------------------------
    def _build_geometry(self):
        n = self.n_positions

        # Fraction of all channels belonging to each stream.
        self.channel_fraction = {
            label: count / n for label, count in self._counts.items()}

        # area_share[label][neighbour] = fraction of ONE label-channel's
        # wetted perimeter that faces `neighbour`.
        #
        # A layer whose two faces meet the SAME stream gives that
        # neighbour its whole perimeter; a layer sandwiched between two
        # DIFFERENT streams splits 50/50. Averaged over that stream's
        # own occurrences so the result is per-channel.
        share = defaultdict(lambda: defaultdict(float))
        for i, label in enumerate(self.layer_order):
            left = self.layer_order[i - 1]                    # wraps at i=0
            right = self.layer_order[(i + 1) % n]
            weight = 1.0 / self._counts[label]
            if left == right:
                share[label][left] += weight
            else:
                share[label][left] += 0.5 * weight
                share[label][right] += 0.5 * weight
        self.area_share = {l: dict(d) for l, d in share.items()}

        # Streams each stream actually exchanges heat with. A stream is
        # never its own neighbour for heat-transfer purposes: two
        # adjacent layers of the SAME stream have no temperature
        # difference across that face, so no duty crosses it.
        self.neighbours = {
            label: sorted(nb for nb in self.area_share[label] if nb != label)
            for label in self.labels}

        # Unique unordered interfaces, as (label_a, label_b) pairs
        # ordered by self.labels so the list is deterministic.
        rank = {label: i for i, label in enumerate(self.labels)}
        pairs = set()
        for label in self.labels:
            for nb in self.neighbours[label]:
                pairs.add(tuple(sorted((label, nb), key=rank.__getitem__)))
        self.interfaces = sorted(
            pairs, key=lambda p: (rank[p[0]], rank[p[1]]))

    # ------------------------------------------------------------------
    # state vector layout
    # ------------------------------------------------------------------
    def _build_state_map(self):
        """Assigns state-vector row indices.

        Per stream: xp (packed-bed streams only - they are the only ones
        whose composition changes), then P, then T. Streams are laid out
        in self.labels order, so the layout is a direct generalisation of
        the hand-written IDX_* constants: for ['c1','h','c2','h'] with
        the hot stream listed first the layout is exactly the original
        [xp, P_h, T_h, P_c1, T_c1, P_c2, T_c2].
        """
        self.index = {}
        idx = 0
        for label in self.labels:
            entry = {}
            if self.streams[label]['type'] == PACKED_BED:
                entry['xp'] = idx
                idx += 1
            else:
                entry['xp'] = None
            entry['P'] = idx
            idx += 1
            entry['T'] = idx
            idx += 1
            self.index[label] = entry
        self.n_states = idx

    # ------------------------------------------------------------------
    # convenience accessors
    # ------------------------------------------------------------------
    def is_packed(self, label):
        return self.streams[label]['type'] == PACKED_BED

    def direction(self, label):
        return self.streams[label]['direction']

    def flow_sign(self, label):
        """+1 for co-current, -1 for counter-current.

        The energy balance is assembled as
            dT/dz = flow_sign * (heat gained by the stream) / (m_dot*cp)
        which reproduces every sign convention derived by hand for the
        existing topologies:
          co-current + losing heat      -> negative dT/dz
          counter-current + gaining heat-> negative dT/dz
          counter-current + losing heat -> POSITIVE dT/dz
        """
        return 1.0 if self.direction(label) == CO_CURRENT else -1.0

    def boundary_side(self, label):
        """'ya' if this stream's inlet is at z=0, 'yb' if at z=L.

        Co-current streams enter at z=0; counter-current streams enter at
        z=L. The BC residual for a stream is therefore taken from
        inlet_properties (ya) or outlet_properties (yb) accordingly.
        """
        return 'ya' if self.direction(label) == CO_CURRENT else 'yb'

    def state_rows(self, label):
        """Row indices this stream owns, in (xp,) P, T order."""
        e = self.index[label]
        rows = [] if e['xp'] is None else [e['xp']]
        return rows + [e['P'], e['T']]

    def packed_labels(self):
        return [l for l in self.labels if self.is_packed(l)]

    def describe(self):
        """Human-readable summary - useful in run logs so a result file
        records exactly which topology produced it."""
        lines = ['-'.join(self.layer_order) + '  (%d states)' % self.n_states]
        for label in self.labels:
            e = self.index[label]
            lines.append(
                '  %-12s %-10s %-16s frac=%.3f  neighbours=%s  rows(xp,P,T)=%s'
                % (label,
                   self.streams[label]['type'],
                   self.streams[label]['direction'],
                   self.channel_fraction[label],
                   ','.join(self.neighbours[label]) or '-',
                   (e['xp'], e['P'], e['T'])))
        for a, b in self.interfaces:
            lines.append('  interface %s<->%s  area_share[%s][%s]=%.3f'
                         % (a, b, a, b, self.area_share[a][b]))
        return '\n'.join(lines)


def from_configuration(configuration):
    """Builds a Topology from a parsed YAML configuration.

    Expects:
        layer_order: [coolant_1, hot, coolant_2, hot]
        streams:
            hot:       {type: packed_bed, direction: co_current, ...}
            coolant_1: {type: plain_fin,  direction: counter_current, ...}
            coolant_2: {type: plain_fin,  direction: counter_current, ...}
    """
    if 'layer_order' not in configuration:
        raise KeyError(
            "configuration has no 'layer_order' - this is the ordered list "
            "of one repeating period of layers, e.g. "
            "[coolant_1, hot, coolant_2, hot]")
    if 'streams' not in configuration:
        raise KeyError(
            "configuration has no 'streams' block defining each label's "
            "type and direction")
    return Topology(configuration['layer_order'], configuration['streams'])
