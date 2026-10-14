"""Forward-mode automatic differentiation for the analytic Jacobian.

A Dual carries a value and its gradient with respect to one node's state
vector. The model code (hexs, heat_transfer_models, ortho_para_dynamics) is
evaluated unchanged on dual numbers: install_shim replaces those modules'
numpy with a shim whose exp, log, sqrt, tanh, sum and mean also accept
duals, so every result carries its exact derivatives.
"""


import numpy as _real_np


class Dual:
    """Number with value `val` and gradient `der` (array over the node's states)."""
    # arithmetic: value and gradient by the usual differentiation rules
    __slots__ = ('val', 'der')

    def __init__(self, val, der):
        self.val = val
        self.der = _real_np.asarray(der, dtype=float)

    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val + other.val, self.der + other.der)
        return Dual(self.val + other, self.der)
    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val - other.val, self.der - other.der)
        return Dual(self.val - other, self.der)

    def __rsub__(self, other):
        return Dual(other - self.val, -self.der)

    def __mul__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val * other.val, self.der * other.val + self.val * other.der)
        return Dual(self.val * other, self.der * other)
    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val / other.val,
                        (self.der * other.val - self.val * other.der) / (other.val ** 2))
        return Dual(self.val / other, self.der / other)

    def __rtruediv__(self, other):
        return Dual(other / self.val, -other * self.der / (self.val ** 2))

    def __neg__(self):
        return Dual(-self.val, -self.der)

    def __pow__(self, p):
        if isinstance(p, Dual):
            raise NotImplementedError("Dual exponent not needed/supported in this codebase")
        return Dual(self.val ** p, p * self.val ** (p - 1) * self.der)

    def __rpow__(self, base):
        if isinstance(base, Dual):
            raise NotImplementedError("Dual base with Dual exponent not needed here")
        val = base ** self.val
        return Dual(val, val * _real_np.log(base) * self.der)

    # comparisons act on the value only (used by clamps and branches in the model)
    def __lt__(self, other): return self.val < (other.val if isinstance(other, Dual) else other)
    def __le__(self, other): return self.val <= (other.val if isinstance(other, Dual) else other)
    def __gt__(self, other): return self.val > (other.val if isinstance(other, Dual) else other)
    def __ge__(self, other): return self.val >= (other.val if isinstance(other, Dual) else other)
    def __eq__(self, other): return self.val == (other.val if isinstance(other, Dual) else other)
    __hash__ = None

    def __abs__(self):
        sign = 1.0 if self.val >= 0 else -1.0
        return Dual(abs(self.val), sign * self.der)

    def __repr__(self):
        return f"Dual({self.val!r}, der_nonzero={_real_np.nonzero(self.der)[0].tolist()})"

    # numpy calls these methods when a ufunc receives an object
    def exp(self):
        return _dexp(self)

    def log(self):
        return _dlog(self)

    def sqrt(self):
        return _dsqrt(self)

    def tanh(self):
        return _dtanh(self)


def _dexp(x):
    """exp for floats, arrays and duals."""
    if isinstance(x, Dual):
        v = _real_np.exp(x.val)
        return Dual(v, v * x.der)
    return _real_np.exp(x)


def _dlog(x):
    """Natural log for floats, arrays and duals."""
    if isinstance(x, Dual):
        return Dual(_real_np.log(x.val), x.der / x.val)
    return _real_np.log(x)


def _dsqrt(x):
    """Square root for floats, arrays and duals."""
    if isinstance(x, Dual):
        v = _real_np.sqrt(x.val)
        return Dual(v, 0.5 / v * x.der)
    return _real_np.sqrt(x)


def _dtanh(x):
    """tanh for floats, arrays and duals."""
    if isinstance(x, Dual):
        v = _real_np.tanh(x.val)
        return Dual(v, (1 - v ** 2) * x.der)
    return _real_np.tanh(x)


def _dmean(arr):
    """Mean of a sequence that may contain duals."""
    items = list(arr) if hasattr(arr, '__iter__') else [arr]
    total = items[0]
    for a in items[1:]:
        total = total + a
    return total / len(items)


def _dsum(arr):
    """Sum of a sequence that may contain duals."""
    items = list(arr)
    total = items[0]
    for a in items[1:]:
        total = total + a
    return total


class NumpyShim:

    """Stand-in for numpy inside the shimmed modules: dual-aware elementary
    functions, everything else forwarded to numpy.
    """
    def __init__(self):
        self.exp = _dexp
        self.log = _dlog
        self.sqrt = _dsqrt
        self.tanh = _dtanh
        self.mean = _dmean
        self.sum = _dsum

    def __getattr__(self, name):
        return getattr(_real_np, name)


# single shared shim instance
SHIM = NumpyShim()


def install_shim(*modules):
    """Replace each module's `np` by the shim (done once when analytic_jacobian
    is imported). Returns the original numpy references.
    """
    originals = {}
    for m in modules:
        originals[m] = getattr(m, 'np', None)
        m.np = SHIM
    return originals
