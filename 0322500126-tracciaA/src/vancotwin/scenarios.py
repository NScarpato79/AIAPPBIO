"""Simulazione di scenari terapeutici con quantificazione dell'incertezza.

Per ogni regime candidato (dose x intervallo) il Digital Twin propaga *tutti* i
campioni dei parametri individuali (posterior PyMC o ensemble EnKF) e calcola la
distribuzione predittiva di AUC24 e concentrazione di valle. Da qui:
* PTA  = P(400 <= AUC24 <= 600)  (probabilita' di raggiungere il target)
* P(AUC24 > 600)                 (rischio di nefrotossicita')
Il regime raccomandato massimizza la PTA; a parita', minimizza il rischio e la dose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from .calibration import ParameterSamples
from .config import TherapeuticTarget
from .pk_model import Dose, concentration, regimen


@dataclass(frozen=True)
class Regimen:
    amount: float
    interval: float

    @property
    def daily_dose(self) -> float:
        return self.amount * 24.0 / self.interval

    @property
    def label(self) -> str:
        return f"{int(self.amount)} mg q{int(self.interval)}h"


def candidate_regimens(max_daily: float = 6000.0) -> list[Regimen]:
    out = []
    for interval in (8.0, 12.0, 24.0):
        for amount in np.arange(500.0, 2750.0, 250.0):
            r = Regimen(float(amount), interval)
            if r.daily_dose <= max_daily:
                out.append(r)
    return out


def apply_regimen(past_doses: Sequence[Dose], reg: Regimen, t_start: float, horizon: float = 72.0) -> list[Dose]:
    """Mantiene le dosi gia' iniziate prima di ``t_start`` e applica il nuovo regime da li' in poi."""
    kept = [d for d in past_doses if d.time < t_start]
    n = int(np.ceil(horizon / reg.interval)) + 1
    return kept + regimen(reg.amount, reg.interval, n, start=t_start)


def exposure(samples: ParameterSamples, doses: Sequence[Dose], t0: float, t1: float, n_grid: int = 241) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(t0, t1, n_grid)
    conc = concentration(grid, doses, samples.cl, samples.v)
    return trapezoid(conc, grid, axis=-1), conc.min(axis=-1)


def evaluate_regimen(
    samples: ParameterSamples,
    past_doses: Sequence[Dose],
    reg: Regimen,
    t_dec: float,
    target: TherapeuticTarget = TherapeuticTarget(),
) -> dict:
    doses = apply_regimen(past_doses, reg, t_dec)
    auc24, cmin = exposure(samples, doses, t_dec + 24.0, t_dec + 48.0)
    auc_mic = auc24 / target.mic
    return {
        "regimen": reg.label,
        "amount": reg.amount,
        "interval": reg.interval,
        "daily_dose": reg.daily_dose,
        "auc_median": float(np.median(auc_mic)),
        "auc_q05": float(np.quantile(auc_mic, 0.05)),
        "auc_q95": float(np.quantile(auc_mic, 0.95)),
        "pta": float(np.mean((auc_mic >= target.auc_low) & (auc_mic <= target.auc_high))),
        "p_auc_toxic": float(np.mean(auc_mic > target.auc_high)),
        "p_trough_toxic": float(np.mean(cmin > target.trough_toxic)),
    }


def recommend(
    samples: ParameterSamples,
    past_doses: Sequence[Dose],
    t_dec: float,
    target: TherapeuticTarget = TherapeuticTarget(),
    candidates: Sequence[Regimen] | None = None,
    max_samples: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[Regimen, pd.DataFrame]:
    if samples.cl.size > max_samples:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(samples.cl.size, max_samples, replace=False)
        samples = ParameterSamples(samples.cl[idx], samples.v[idx], samples.label)
    candidates = list(candidates or candidate_regimens())
    table = pd.DataFrame([evaluate_regimen(samples, past_doses, r, t_dec, target) for r in candidates])
    table["_pta_r"] = table["pta"].round(2)
    table = table.sort_values(["_pta_r", "p_auc_toxic", "daily_dose"], ascending=[False, True, True]).drop(columns="_pta_r")
    best = table.iloc[0]
    return Regimen(float(best["amount"]), float(best["interval"])), table.reset_index(drop=True)


def concentration_bands(samples: ParameterSamples, doses: Sequence[Dose], t_grid, q=(0.05, 0.5, 0.95)) -> np.ndarray:
    """Quantili (bande di incertezza predittiva) della concentrazione nel tempo."""
    conc = concentration(t_grid, doses, samples.cl, samples.v)
    return np.quantile(conc, q, axis=0)
