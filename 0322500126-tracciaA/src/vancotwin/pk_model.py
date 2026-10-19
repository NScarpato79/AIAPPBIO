"""Modello farmacocinetico monocompartimentale con infusione endovenosa intermittente.

Dinamica (ODE):
    dC/dt = R(t) / V - (CL / V) * C
dove R(t) e' la velocita' di infusione [mg/h], CL la clearance [L/h], V il volume [L].

Sono fornite due implementazioni equivalenti:
* ``simulate_ode``: integrazione numerica (scipy) vettorizzata su un ensemble,
  usata dall'EnKF che propaga lo stato C nel tempo;
* ``concentration``: soluzione analitica esatta per sovrapposizione delle dosi,
  differenziabile e quindi utilizzabile anche dentro PyMC (``xp=pytensor.tensor``).
I test verificano che le due coincidano.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.integrate import solve_ivp, trapezoid


@dataclass(frozen=True)
class Dose:
    time: float
    amount: float
    duration: float = 1.0

    @property
    def rate(self) -> float:
        return self.amount / self.duration

    @property
    def end(self) -> float:
        return self.time + self.duration


def infusion_duration(amount: float) -> float:
    """Durata di infusione: 1 h ogni 1000 mg (minimo 1 h), per limitare la red man syndrome."""
    return max(1.0, amount / 1000.0)


def regimen(
    amount: float,
    interval: float,
    n_doses: int,
    start: float = 0.0,
    duration: float | None = None,
) -> list[Dose]:
    dur = infusion_duration(amount) if duration is None else duration
    return [Dose(start + i * interval, amount, dur) for i in range(n_doses)]


def infusion_rate(t: float, doses: Sequence[Dose]) -> float:
    return float(sum(d.rate for d in doses if d.time <= t < d.end))


def _superposition(t, doses: Sequence[Dose], cl, v, xp):
    k = cl / v
    c = 0.0
    for d in doses:
        tau = t - d.time
        t_in = xp.clip(tau, 0.0, d.duration)
        t_post = xp.maximum(tau - d.duration, 0.0)
        c = c + d.rate / cl * (1.0 - xp.exp(-k * t_in)) * xp.exp(-k * t_post)
    return c


def concentration(t, doses: Sequence[Dose], cl, v) -> np.ndarray:
    """Concentrazione analitica [mg/L].

    ``t`` ha forma (T,), ``cl`` e ``v`` scalari o array (N,): il risultato ha forma
    (T,) oppure (N, T).
    """
    t = np.asarray(t, dtype=float)
    cl_a = np.asarray(cl, dtype=float)
    v_a = np.asarray(v, dtype=float)
    if cl_a.ndim == 0 and v_a.ndim == 0:
        return np.asarray(_superposition(t, doses, cl_a, v_a, np)) * np.ones_like(t)
    cl_a, v_a = np.broadcast_arrays(cl_a, v_a)
    out = _superposition(t[None, :], doses, cl_a[:, None], v_a[:, None], np)
    return np.asarray(out) * np.ones((cl_a.shape[0], t.shape[0]))


def concentration_tensor(t, doses: Sequence[Dose], cl, v):
    """Versione simbolica (pytensor) della soluzione analitica, per PyMC."""
    import pytensor.tensor as pt

    return _superposition(np.asarray(t, dtype=float), doses, cl, v, pt)


def simulate_ode(
    t_eval,
    doses: Sequence[Dose],
    cl,
    v,
    c0=0.0,
    t0: float = 0.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> np.ndarray:
    """Integra l'ODE da ``t0`` (stato ``c0``) ai tempi ``t_eval`` (>= t0).

    L'integrazione e' spezzata nei punti di discontinuita' di R(t) (inizio/fine
    infusione), cosi' che in ogni segmento il termine forzante sia costante.
    Restituisce un array (N, T) per un ensemble di N membri.
    """
    t_eval = np.atleast_1d(np.asarray(t_eval, dtype=float))
    if np.any(t_eval < t0 - 1e-12):
        raise ValueError("t_eval deve essere >= t0")
    cl_a = np.atleast_1d(np.asarray(cl, dtype=float))
    v_a = np.atleast_1d(np.asarray(v, dtype=float))
    cl_a, v_a = np.broadcast_arrays(cl_a, v_a)
    n = cl_a.shape[0]
    c = np.broadcast_to(np.asarray(c0, dtype=float), (n,)).astype(float).copy()
    k = cl_a / v_a

    t_end = float(t_eval.max()) if t_eval.size else t0
    bps = {t0, t_end}
    for d in doses:
        for b in (d.time, d.end):
            if t0 < b < t_end:
                bps.add(b)
    edges = sorted(bps)

    out = np.empty((n, t_eval.shape[0]))
    out[:, t_eval <= t0] = c[:, None]
    for a, b in zip(edges[:-1], edges[1:]):
        rate = infusion_rate(0.5 * (a + b), doses)
        mask = (t_eval > a) & (t_eval <= b)
        sol = solve_ivp(
            lambda _t, y: rate / v_a - k * y,
            (a, b),
            c,
            t_eval=np.unique(np.append(t_eval[mask], b)),
            rtol=rtol,
            atol=atol,
            vectorized=False,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        if mask.any():
            idx = np.searchsorted(sol.t, t_eval[mask])
            out[:, mask] = sol.y[:, idx]
        c = sol.y[:, -1]
    return out


def auc(doses: Sequence[Dose], cl, v, t_start: float, t_end: float, n_grid: int = 1441) -> np.ndarray:
    """AUC [mg*h/L] della concentrazione su [t_start, t_end]."""
    grid = np.linspace(t_start, t_end, n_grid)
    return trapezoid(concentration(grid, doses, cl, v), grid, axis=-1)


def auc24_steady_state(daily_dose: float, cl) -> np.ndarray:
    """A stato stazionario AUC24 = dose giornaliera / CL (relazione esatta, lineare)."""
    return daily_dose / np.asarray(cl, dtype=float)
