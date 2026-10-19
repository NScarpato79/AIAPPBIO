"""Ensemble Kalman Filter (EnKF) per l'assimilazione dei dati in tempo reale.

Stato aumentato di ogni membro dell'ensemble:  x = [C, log CL, log V].
* Forecast: C viene propagato integrando l'ODE fino all'istante della nuova misura
  (i parametri restano costanti, con una piccola inflazione per evitare il collasso).
* Analysis: aggiornamento stocastico con osservazioni perturbate,
      K = P_xy / (P_yy + R),   x_j <- x_j + K (y + eps_j - H x_j)
  dove H estrae C e R e' la varianza dell'errore di misura.
I parametri sono aggiornati tramite la loro correlazione campionaria con C: e' cosi'
che il filtro "impara" CL e V del paziente man mano che arrivano i prelievi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .calibration import ParameterSamples
from .config import PopulationModel
from .pk_model import Dose, simulate_ode


@dataclass
class EnKFRecord:
    t: float
    y: float | None
    c_mean: float
    cl_mean: float
    cl_q05: float
    cl_q95: float
    v_mean: float
    v_q05: float
    v_q95: float


@dataclass
class EnsembleKalmanFilter:
    doses: Sequence[Dose]
    log_cl: np.ndarray
    log_v: np.ndarray
    rng: np.random.Generator
    pop: PopulationModel = PopulationModel()
    inflation: float = 1.02
    t: float = 0.0
    c: np.ndarray | None = None
    history: list[EnKFRecord] = field(default_factory=list)

    @classmethod
    def from_samples(cls, samples: ParameterSamples, doses, rng, pop=PopulationModel(), n: int | None = None, **kw):
        n = n or samples.cl.size
        idx = rng.choice(samples.cl.size, size=n, replace=samples.cl.size < n)
        return cls(doses, np.log(samples.cl[idx]), np.log(samples.v[idx]), rng, pop, **kw)

    def __post_init__(self):
        self.log_cl = np.asarray(self.log_cl, dtype=float).copy()
        self.log_v = np.asarray(self.log_v, dtype=float).copy()
        if self.c is None:
            self.c = np.zeros_like(self.log_cl)
        self._record(None)

    @property
    def n(self) -> int:
        return self.log_cl.size

    def samples(self) -> ParameterSamples:
        return ParameterSamples(np.exp(self.log_cl), np.exp(self.log_v), "enkf")

    def forecast(self, t_new: float) -> None:
        if t_new < self.t:
            raise ValueError("Le osservazioni devono arrivare in ordine temporale")
        if t_new > self.t:
            self.c = simulate_ode([t_new], self.doses, np.exp(self.log_cl), np.exp(self.log_v), self.c, self.t)[:, 0]
            self.t = t_new

    def update(self, y: float) -> None:
        X = np.vstack([self.c, self.log_cl, self.log_v])
        mean = X.mean(axis=1, keepdims=True)
        # inflazione moltiplicativa delle anomalie dei parametri
        X[1:] = mean[1:] + self.inflation * (X[1:] - mean[1:])
        hx = X[0]
        A = X - X.mean(axis=1, keepdims=True)
        ha = hx - hx.mean()
        p_xy = A @ ha / (self.n - 1)
        p_yy = ha @ ha / (self.n - 1)
        r = float(self.pop.obs_sd(y) ** 2)
        gain = p_xy / (p_yy + r)
        y_pert = y + self.rng.normal(0.0, np.sqrt(r), self.n)
        X = X + gain[:, None] * (y_pert - hx)[None, :]
        self.c = np.maximum(X[0], 0.0)
        self.log_cl, self.log_v = X[1], X[2]

    def assimilate(self, t: float, y: float) -> None:
        """Ciclo forecast + analysis per una nuova misura (ignorata se mancante)."""
        self.forecast(t)
        if np.isfinite(y):
            self.update(float(y))
        self._record(float(y) if np.isfinite(y) else None)

    def _record(self, y):
        cl, v = np.exp(self.log_cl), np.exp(self.log_v)
        self.history.append(
            EnKFRecord(
                self.t, y, float(np.mean(self.c)),
                float(cl.mean()), float(np.quantile(cl, 0.05)), float(np.quantile(cl, 0.95)),
                float(v.mean()), float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95)),
            )
        )

    def predict(self, t_grid) -> np.ndarray:
        """Predizione d'ensemble (N, T) dallo stato corrente per tempi >= t."""
        return simulate_ode(t_grid, self.doses, np.exp(self.log_cl), np.exp(self.log_v), self.c, self.t)
