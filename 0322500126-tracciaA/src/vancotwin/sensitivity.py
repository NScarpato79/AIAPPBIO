"""Spiegabilita' del Digital Twin: analisi di sensitivita' globale (indici di Sobol).

Il modello e' meccanicistico (white-box), quindi la spiegazione si ottiene
scomponendo la varianza dell'output (AUC24, valle) tra i fattori di input:
covariate cliniche (CrCl, peso) e variabilita' individuale non spiegata (eta).
Stimatori: Saltelli (2010) per S1, Jansen per ST.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .config import PopulationModel

Sampler = Callable[[int, np.random.Generator], np.ndarray]


def sobol_indices(func: Callable[[dict], np.ndarray], samplers: dict[str, Sampler], n: int, rng: np.random.Generator) -> pd.DataFrame:
    names = list(samplers)
    A = {k: s(n, rng) for k, s in samplers.items()}
    B = {k: s(n, rng) for k, s in samplers.items()}
    f_a, f_b = func(A), func(B)
    var = np.var(np.concatenate([f_a, f_b]), ddof=1)
    rows = []
    for name in names:
        ab = dict(A)
        ab[name] = B[name]
        f_ab = func(ab)
        s1 = np.mean(f_b * (f_ab - f_a)) / var
        st = 0.5 * np.mean((f_a - f_ab) ** 2) / var
        rows.append({"factor": name, "S1": float(s1), "ST": float(st)})
    return pd.DataFrame(rows)


def trough_steady_state(amount: float, interval: float, duration: float, cl, v) -> np.ndarray:
    """Valle a stato stazionario per infusione intermittente (modello monocompartimentale)."""
    cl, v = np.asarray(cl, float), np.asarray(v, float)
    k = cl / v
    rate = amount / duration
    return rate / cl * (1 - np.exp(-k * duration)) * np.exp(-k * (interval - duration)) / (1 - np.exp(-k * interval))


def population_sensitivity(n: int, rng: np.random.Generator, pop: PopulationModel = PopulationModel(), amount: float = 1000.0, interval: float = 12.0) -> pd.DataFrame:
    """Sobol di AUC24 e valle a stato stazionario per un regime fisso standard."""
    samplers = {
        "CrCl": lambda m, r: r.uniform(15, 150, m),
        "peso": lambda m, r: np.clip(r.lognormal(np.log(72), 0.2, m), 40, 150),
        "eta_CL": lambda m, r: r.normal(0, pop.omega_cl, m),
        "eta_V": lambda m, r: r.normal(0, pop.omega_v, m),
    }

    def params(x):
        return pop.cl_pop(x["CrCl"]) * np.exp(x["eta_CL"]), pop.v_pop(x["peso"]) * np.exp(x["eta_V"])

    def f_auc(x):
        cl, _ = params(x)
        return amount * 24.0 / interval / cl

    def f_trough(x):
        cl, v = params(x)
        return trough_steady_state(amount, interval, 1.0, cl, v)

    out = []
    for label, f in (("AUC24", f_auc), ("Cvalle", f_trough)):
        df = sobol_indices(f, samplers, n, rng)
        df.insert(0, "output", label)
        out.append(df)
    return pd.concat(out, ignore_index=True)


def posterior_shrinkage(prior_sd: float, posterior_sd: float) -> float:
    """1 - var_post/var_prior: quanto i dati del paziente hanno ridotto l'incertezza."""
    return float(1.0 - (posterior_sd / prior_sd) ** 2)
