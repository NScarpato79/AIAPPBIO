"""Calibrazione bayesiana dei parametri individuali con PyMC.

Modello gerarchico "a priori di popolazione":
    eta_CL, eta_V ~ N(0, 1)                         (parametrizzazione non centrata)
    CL = CL_pop(CrCl) * exp(omega_CL * eta_CL)
    V  = V_pop(peso)  * exp(omega_V  * eta_V)
    C_oss ~ N(C(t; CL, V), sqrt(sigma_add^2 + (sigma_prop * C)^2))
C(t) e' la soluzione esatta dell'ODE (sovrapposizione analitica), espressa in
pytensor e quindi differenziabile: NUTS puo' usare i gradienti.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import PopulationModel
from .pk_model import Dose, concentration_tensor


@dataclass
class ParameterSamples:
    cl: np.ndarray
    v: np.ndarray
    label: str
    diagnostics: dict | None = None

    def summary(self) -> dict:
        def q(x):
            return {"mean": float(np.mean(x)), "q05": float(np.quantile(x, 0.05)), "q95": float(np.quantile(x, 0.95))}

        out = {"CL": q(self.cl), "V": q(self.v), "n": int(self.cl.size)}
        if self.diagnostics:
            out["diagnostics"] = self.diagnostics
        return out


def prior_samples(crcl: float, weight: float, n: int, rng: np.random.Generator, pop: PopulationModel = PopulationModel()) -> ParameterSamples:
    """Campioni dall'a priori di popolazione (Digital Twin "a priori", prima di ogni misura)."""
    cl = pop.cl_pop(crcl) * np.exp(rng.normal(0.0, pop.omega_cl, n))
    v = pop.v_pop(weight) * np.exp(rng.normal(0.0, pop.omega_v, n))
    return ParameterSamples(cl, v, "population_prior")


def _valid(t_obs, c_obs):
    t_obs = np.asarray(t_obs, dtype=float)
    c_obs = np.asarray(c_obs, dtype=float)
    keep = np.isfinite(c_obs)
    return t_obs[keep], c_obs[keep]


def build_model(crcl: float, weight: float, doses: Sequence[Dose], t_obs, c_obs, pop: PopulationModel = PopulationModel()):
    import pymc as pm
    import pytensor.tensor as pt

    t_obs, c_obs = _valid(t_obs, c_obs)
    with pm.Model() as model:
        eta_cl = pm.Normal("eta_cl", 0.0, 1.0)
        eta_v = pm.Normal("eta_v", 0.0, 1.0)
        cl = pm.Deterministic("CL", float(pop.cl_pop(crcl)) * pt.exp(pop.omega_cl * eta_cl))
        v = pm.Deterministic("V", float(pop.v_pop(weight)) * pt.exp(pop.omega_v * eta_v))
        if t_obs.size:
            mu = concentration_tensor(t_obs, doses, cl, v)
            sd = pt.sqrt(pop.sigma_add**2 + (pop.sigma_prop * mu) ** 2)
            pm.Normal("obs", mu=mu, sigma=sd, observed=c_obs)
    return model


def fit_posterior(
    crcl: float,
    weight: float,
    doses: Sequence[Dose],
    t_obs,
    c_obs,
    pop: PopulationModel = PopulationModel(),
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    seed: int = 0,
) -> ParameterSamples:
    """Campiona il posterior individuale con NUTS e restituisce CL, V e diagnostiche."""
    import arviz as az
    import pymc as pm

    model = build_model(crcl, weight, doses, t_obs, c_obs, pop)
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            target_accept=0.9,
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=False,
        )
    post = idata.posterior
    summ = az.summary(idata, var_names=["CL", "V"], kind="diagnostics")
    diagnostics = {
        "r_hat_max": float(summ["r_hat"].max()) if chains > 1 else float("nan"),
        "ess_bulk_min": float(summ["ess_bulk"].min()),
        "divergences": int(idata.sample_stats["diverging"].sum()),
    }
    return ParameterSamples(
        post["CL"].values.reshape(-1),
        post["V"].values.reshape(-1),
        "bayes_pymc",
        diagnostics,
    )
