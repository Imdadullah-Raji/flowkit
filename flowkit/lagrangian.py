"""
Particle advection through a velocity field.

The central rule: **the particle array never changes length.** A particle that
leaves the domain becomes NaN and stays NaN. Deleting escaped particles, as an
earlier version did, destroys the correspondence between a particle's index and
its seeding position -- which is precisely what a flow map is, so it silently
invalidated every FTLE calculation built on it.
"""

import numpy as np

from flowkit.dataprocessing import CartesianSnapshot, Dataset


class FlowField:
    """
    Velocity sampled at any (x, y, t), interpolated between snapshots.

    A `CartesianSnapshot` is frozen in time, so integrating through one with a
    dt smaller than the snapshot spacing sees a stationary flow. This blends the
    two bracketing snapshots instead, and caches the (expensive) unstructured to
    Cartesian interpolation so each snapshot is gridded at most once.
    """

    time_dependent = True

    def __init__(self, dataset, element_size=None, cache_size=4):
        self._ds = dataset if isinstance(dataset, Dataset) else Dataset.from_dataset(dataset)
        self._times = np.asarray(self._ds.times, dtype=float)
        if self._times.size < 2:
            raise ValueError("A time-dependent field needs at least 2 snapshots.")
        self._h = element_size
        self._cache = {}
        self._cache_size = max(2, cache_size)

    @property
    def times(self):
        return self._times

    @property
    def t0(self):
        return float(self._times[0])

    @property
    def t1(self):
        return float(self._times[-1])

    def frame(self, idx) -> CartesianSnapshot:
        """The gridded field at snapshot `idx`, cached."""
        idx = int(idx)
        if idx not in self._cache:
            if len(self._cache) >= self._cache_size:
                self._cache.pop(min(self._cache))
            snap = self._ds.snapshot(float(self._times[idx]))
            self._cache[idx] = snap.interpolate_on_grid(element_size=self._h)
        return self._cache[idx]

    def _bracket(self, t):
        """Left snapshot index and the blend weight toward the next one."""
        i = int(np.clip(np.searchsorted(self._times, t, side="right") - 1,
                        0, self._times.size - 2))
        span = self._times[i + 1] - self._times[i]
        w = 0.0 if span <= 0 else float((t - self._times[i]) / span)
        return i, min(max(w, 0.0), 1.0)

    def sample(self, xq, yq, t):
        """Velocity (u, v) at the given points and time. NaN outside the domain."""
        i, w = self._bracket(t)
        u0, v0, _ = self.frame(i).sample(xq, yq)
        if w == 0.0:
            return u0, v0
        u1, v1, _ = self.frame(i + 1).sample(xq, yq)
        return (1.0 - w) * u0 + w * u1, (1.0 - w) * v0 + w * v1

    def __repr__(self):
        return (f"<FlowField {self._times.size} snapshots "
                f"[{self.t0:g}..{self.t1:g}], {len(self._cache)} gridded>")


def _uv(field, x, y, t):
    """Velocity from either a frozen CartesianSnapshot or a time-dependent field."""
    if getattr(field, "time_dependent", False):
        return field.sample(x, y, t)
    u, v, _ = field.sample(x, y)
    return u, v


class Particles:
    """
    A fixed set of tracer positions.

    `posx`/`posy` always have the length they were created with. Escaped
    particles hold NaN, so `posx.reshape(seed_shape)` stays valid for the whole
    integration -- the property the flow map depends on.
    """

    def __init__(self, x, y, mask=None):
        self.posx = np.array(x, dtype=float).ravel()
        self.posy = np.array(y, dtype=float).ravel()
        if self.posx.shape != self.posy.shape:
            raise ValueError(
                f"x and y must match: got {self.posx.shape} and {self.posy.shape}"
            )
        # mask=True marks a particle to exclude (e.g. inside a solid body).
        # It is blanked rather than removed, keeping the array length fixed.
        self._mask = None if mask is None else np.asarray(mask, dtype=bool).ravel()
        if self._mask is not None:
            if self._mask.shape != self.posx.shape:
                raise ValueError("mask must have the same shape as x and y")
            self.posx[self._mask] = np.nan
            self.posy[self._mask] = np.nan

    @classmethod
    def on_grid(cls, x, y, mask=None):
        """Seed on the meshgrid of 1-D axes `x` and `y`. Returns (particles, shape)."""
        X, Y = np.meshgrid(np.asarray(x), np.asarray(y))
        return cls(X, Y, mask=mask), X.shape

    @property
    def excluded(self):
        return self._mask

    @property
    def active(self):
        """Particles still inside the domain."""
        return ~(np.isnan(self.posx) | np.isnan(self.posy))

    @property
    def n_particles(self):
        return self.posx.size

    @property
    def n_active(self):
        return int(self.active.sum())

    def step(self, field, dt, t=None, method="rk4"):
        """
        One integration step. Returns a new Particles of the same length.

        `method` is "rk4" (default) or "euler". A particle that samples NaN --
        because it left the domain -- keeps NaN through the arithmetic and stays
        escaped, which is the intended behaviour.
        """
        if dt is None:
            raise TypeError("dt is required")
        m = str(method).lower()
        if m == "euler":
            x, y = self._euler(field, float(dt), t)
        elif m == "rk4":
            x, y = self._rk4(field, float(dt), t)
        else:
            raise ValueError(f"Unknown method {method!r}; use 'rk4' or 'euler'.")
        out = Particles(x, y)
        out._mask = self._mask
        return out

    def _euler(self, field, dt, t):
        u, v = _uv(field, self.posx, self.posy, t)
        return self.posx + u * dt, self.posy + v * dt

    def _rk4(self, field, dt, t):
        t = 0.0 if t is None else t
        x, y = self.posx, self.posy
        k1u, k1v = _uv(field, x, y, t)
        k2u, k2v = _uv(field, x + 0.5 * dt * k1u, y + 0.5 * dt * k1v, t + 0.5 * dt)
        k3u, k3v = _uv(field, x + 0.5 * dt * k2u, y + 0.5 * dt * k2v, t + 0.5 * dt)
        k4u, k4v = _uv(field, x + dt * k3u, y + dt * k3v, t + dt)
        return (x + dt / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u),
                y + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v))

    def __len__(self):
        return self.posx.size

    def __repr__(self):
        return f"<Particles {self.n_particles} total, {self.n_active} active>"


def advect(field, particles, t0, T, dt, method="rk4", element_size=None):
    """
    Integrate `particles` from t0 over an interval T. Returns a new Particles.

    `field` may be a Dataset (gridded and interpolated in time on the fly), a
    FlowField, or a single CartesianSnapshot for a frozen field. A negative T
    integrates backwards, which is what backward-time FTLE needs.

    The final step is trimmed so the integration lands exactly on t0 + T.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive; use a negative T to go backwards, got {dt!r}")
    if isinstance(field, (Dataset,)) or hasattr(field, "to_dataset"):
        field = FlowField(field, element_size=element_size)

    n = int(np.ceil(abs(T) / dt))
    step = np.sign(T) * dt if T != 0 else 0.0
    t = float(t0)
    for i in range(n):
        h = step
        if abs(t + h - t0) > abs(T):        # trim the last step onto the target
            h = (t0 + T) - t
        particles = particles.step(field, h, t=t, method=method)
        t += h
    return particles
