"""Stream topology of the exchanger: which streams exist, how their layers are
stacked, and how the solver's state vector is laid out.

layer_order lists one repeating period of the stack (e.g. [coolant_1, hot,
coolant_2, hot]); the period is treated as cyclic, so the last layer touches
the first. From it follow each stream's share of the channels, its
neighbours, the heat-exchange interfaces and the heated perimeter facing
each neighbour (used by hexs.PlateFinHex), and the index of every state
variable (used by bvp_model). Small or non-repeating stacks are represented
by mirroring them (e.g. A-B-C as A B C C B A).
"""


from collections import Counter, defaultdict


# stream types: catalyst-filled channels, or serrated (offset-strip) fins without catalyst
PACKED_BED = 'packed_bed'
OFFSET_STRIP = 'offset_strip'
# legacy type name, accepted as an alias of offset_strip (with a deprecation warning)
_LEGACY_TYPE_NAMES = {'plain_fin': OFFSET_STRIP}

# flow directions: co-current enters at z=0, counter-current at z=L
CO_CURRENT = 'co_current'
COUNTER_CURRENT = 'counter_current'

_VALID_TYPES = (PACKED_BED, OFFSET_STRIP)
_VALID_DIRECTIONS = (CO_CURRENT, COUNTER_CURRENT)


class Topology(object):

    """Layer arrangement and state layout, built from the configuration's
    'layer_order' and 'streams' blocks by from_configuration.
    """
    def __init__(self, layer_order, streams):
        self._validate(layer_order, streams)
        self.layer_order = list(layer_order)
        self.streams = streams

        # stream labels in order of first appearance (defines the state order)
        seen = []
        for label in self.layer_order:
            if label not in seen:
                seen.append(label)
        self.labels = seen

        self._counts = Counter(self.layer_order)
        self.n_positions = len(self.layer_order)

        self._build_geometry()
        self._build_state_map()

    @staticmethod
    def _validate(layer_order, streams):
        """Check that layer_order and streams are consistent, map the deprecated
        type name 'plain_fin' to 'offset_strip', and check types and directions.
        """
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
            if stype in _LEGACY_TYPE_NAMES:
                import warnings
                warnings.warn("stream %r: type %r is deprecated, use %r (these "
                              "are offset-strip fins, not plain fins)"
                              % (label, stype, _LEGACY_TYPE_NAMES[stype]))
                stype = _LEGACY_TYPE_NAMES[stype]
                spec['type'] = stype
            if stype not in _VALID_TYPES:
                raise ValueError(
                    "stream %r has type %r; expected one of %s"
                    % (label, stype, list(_VALID_TYPES)))
            direction = spec.get('direction')
            if direction not in _VALID_DIRECTIONS:
                raise ValueError(
                    "stream %r has direction %r; expected one of %s"
                    % (label, direction, list(_VALID_DIRECTIONS)))

        if len(set(layer_order)) < 2:
            raise ValueError(
                "layer_order must contain at least 2 distinct streams; got %r"
                % (layer_order,))

    def _build_geometry(self):
        """Derive the stream-level geometry from the cyclic layer sequence.

        channel_fraction: share of all layers belonging to each stream.
        area_share[a][b]: share of a's perimeter in CONTACT with b (half per wall
          for a layer between two different streams); defines the interfaces.
        heated_share_two / heated_share_one: share of a's perimeter that heats
          towards b, split into two-sided layers (heated through both walls,
          half-height fins) and one-sided layers (one wall faces a layer of the
          same stream, e.g. double stacking or a mirror plane: the whole perimeter
          faces the one neighbour, with full-height fins).
        neighbours / interfaces: stream pairs that exchange heat.
        """
        n = self.n_positions

        self.channel_fraction = {
            label: count / n for label, count in self._counts.items()}

        share = defaultdict(lambda: defaultdict(float))
        share_two = defaultdict(lambda: defaultdict(float))
        share_one = defaultdict(lambda: defaultdict(float))
        # walk the cyclic sequence; each layer carries 1/count of its stream
        for i, label in enumerate(self.layer_order):
            left = self.layer_order[i - 1]
            right = self.layer_order[(i + 1) % n]
            weight = 1.0 / self._counts[label]
            # both walls face the same stream: whole perimeter towards it, heated from both walls
            if left == right:
                share[label][left] += weight
                share_two[label][left] += weight
            else:
                share[label][left] += 0.5 * weight
                share[label][right] += 0.5 * weight
                # one-sided layer: one wall faces the same stream, the other a different stream
                if label in (left, right):
                    other = right if left == label else left
                    share_one[label][other] += weight
                else:
                    share_two[label][left] += 0.5 * weight
                    share_two[label][right] += 0.5 * weight
        self.area_share = {l: dict(d) for l, d in share.items()}
        self.heated_share_two = {l: dict(d) for l, d in share_two.items()}
        self.heated_share_one = {l: dict(d) for l, d in share_one.items()}

        # neighbours exclude the stream itself (same-stream contact exchanges nothing)
        self.neighbours = {
            label: sorted(nb for nb in self.area_share[label] if nb != label)
            for label in self.labels}

        # unique interface pairs, ordered by first appearance of the streams
        rank = {label: i for i, label in enumerate(self.labels)}
        pairs = set()
        for label in self.labels:
            for nb in self.neighbours[label]:
                pairs.add(tuple(sorted((label, nb), key=rank.__getitem__)))
        self.interfaces = sorted(
            pairs, key=lambda p: (rank[p[0]], rank[p[1]]))

    def _build_state_map(self):
        """State-vector layout: per stream (in order of first appearance) a para
        fraction (catalysed streams only), a pressure (kPa) and a temperature (K).
        index[label] gives the row of each; n_states is the total.
        """
        self.index = {}
        idx = 0
        for label in self.labels:
            entry = {}
            # only catalysed streams have a para-fraction state
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

    def is_packed(self, label):
        """True for catalyst-filled (packed-bed) streams, which carry a para fraction."""
        return self.streams[label]['type'] == PACKED_BED

    def direction(self, label):
        """'co_current' (flows towards +z) or 'counter_current' (towards -z)."""
        return self.streams[label]['direction']

    def flow_sign(self, label):
        """+1 for co-current, -1 for counter-current: converts per-length changes along
        the flow into d/dz.
        """
        return 1.0 if self.direction(label) == CO_CURRENT else -1.0

    def boundary_side(self, label):
        """End where the stream enters: 'ya' (z=0) for co-current, 'yb' (z=L) otherwise."""
        return 'ya' if self.direction(label) == CO_CURRENT else 'yb'

    def state_rows(self, label):
        """State rows pinned by this stream's inlet boundary condition."""
        e = self.index[label]
        rows = [] if e['xp'] is None else [e['xp']]
        return rows + [e['P'], e['T']]

    def packed_labels(self):
        """Labels of all catalysed streams."""
        return [l for l in self.labels if self.is_packed(l)]

    def describe(self):
        """Readable summary of the topology (for printing and debugging)."""
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
    """Build the Topology from a configuration dict (entry point used by model.model)."""
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
