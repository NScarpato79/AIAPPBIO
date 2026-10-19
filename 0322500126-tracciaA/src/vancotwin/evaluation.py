"""Validazione del Digital Twin sulla coorte virtuale (con ground truth nota).

Protocollo per ogni paziente:
1. si usano i primi due prelievi (picco ~14 h, valle ~35.5 h) per personalizzare il twin;
2. si predice il prelievo *tenuto da parte* (valle ~71.5 h) e l'AUC24 vera 48-72 h;
3. alla decisione (36 h) il twin raccomanda un regime e si misura l'AUC "vera"
   ottenuta dal paziente virtuale, confrontandola con la prosecuzione del regime standard.
Metodi confrontati: a priori di popolazione, posterior bayesiano (PyMC), EnKF.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import ParameterSamples, fit_posterior, prior_samples
from .config import PopulationModel, RunConfig, TherapeuticTarget
from .data import Patient
from .enkf import EnsembleKalmanFilter
from .pk_model import auc, concentration
from .scenarios import Regimen, apply_regimen, recommend


def _true_auc(p: Patient, doses, t0: float, t1: float) -> float:
    return float(auc(doses, p.cl_true, p.v_true, t0, t1))


def personalize(p: Patient, cfg: RunConfig, pop: PopulationModel, rng: np.random.Generator, seed: int):
    """Restituisce i tre "twin" del paziente: prior, PyMC, EnKF (calibrati sui primi 2 prelievi)."""
    t_fit, c_fit = p.obs_times[:2], p.obs_conc[:2]
    prior = prior_samples(p.crcl, p.weight, 2000, rng, pop)
    bayes = fit_posterior(p.crcl, p.weight, p.doses, t_fit, c_fit, pop, cfg.draws, cfg.tune, cfg.chains, seed)
    enkf = EnsembleKalmanFilter.from_samples(prior, p.doses, rng, pop, n=cfg.n_ensemble)
    for t, y in zip(t_fit, c_fit):
        enkf.assimilate(float(t), float(y))
    return prior, bayes, enkf


def evaluate_patient(p: Patient, cfg: RunConfig, pop: PopulationModel, target: TherapeuticTarget, rng: np.random.Generator, seed: int) -> list[dict]:
    prior, bayes, enkf = personalize(p, cfg, pop, rng, seed)
    t_ho, c_ho = float(p.obs_times[p.holdout_index]), float(p.obs_conc[p.holdout_index])
    t_dec = cfg.decision_time
    auc_true = _true_auc(p, p.doses, 48.0, 72.0)
    standard_auc = _true_auc(p, p.doses, t_dec + 24.0, t_dec + 48.0)

    rows = []
    for name, samples in (("population_prior", prior), ("bayes_pymc", bayes), ("enkf", enkf.samples())):
        if name == "enkf":
            pred = enkf.predict([t_ho])[:, 0]
        else:
            pred = concentration([t_ho], p.doses, samples.cl, samples.v)[:, 0]
        auc_pred = auc(p.doses, samples.cl, samples.v, 48.0, 72.0, n_grid=241)
        reg, _ = recommend(samples, p.doses, t_dec, target, rng=rng)
        achieved = _true_auc(p, apply_regimen(p.doses, reg, t_dec), t_dec + 24.0, t_dec + 48.0)
        rows.append(
            {
                "pid": p.pid,
                "method": name,
                "sex": p.sex,
                "age": p.age,
                "crcl": p.crcl,
                "n_fit_obs": int(np.isfinite(p.obs_conc[:2]).sum()),
                "obs_holdout": c_ho,
                "pred_median": float(np.median(pred)),
                "pred_q05": float(np.quantile(pred, 0.05)),
                "pred_q95": float(np.quantile(pred, 0.95)),
                "auc_true": auc_true,
                "auc_pred_median": float(np.median(auc_pred)),
                "auc_pred_q05": float(np.quantile(auc_pred, 0.05)),
                "auc_pred_q95": float(np.quantile(auc_pred, 0.95)),
                "recommended": reg.label,
                "recommended_daily": reg.daily_dose,
                "achieved_auc": achieved,
                "standard_auc": standard_auc,
                "r_hat_max": (bayes.diagnostics or {}).get("r_hat_max") if name == "bayes_pymc" else None,
                "divergences": (bayes.diagnostics or {}).get("divergences") if name == "bayes_pymc" else None,
            }
        )
    return rows


def _metrics(g: pd.DataFrame, target: TherapeuticTarget) -> dict:
    err = g["pred_median"] - g["obs_holdout"]
    auc_err = g["auc_pred_median"] - g["auc_true"]
    in_t = lambda x: (x >= target.auc_low) & (x <= target.auc_high)  # noqa: E731
    return {
        "n": int(len(g)),
        "conc_MPE_pct": float(np.mean(err / g["obs_holdout"]) * 100),
        "conc_MAPE_pct": float(np.mean(np.abs(err) / g["obs_holdout"]) * 100),
        "conc_RMSE": float(np.sqrt(np.mean(err**2))),
        "conc_PI90_coverage": float(np.mean((g["obs_holdout"] >= g["pred_q05"]) & (g["obs_holdout"] <= g["pred_q95"]))),
        "auc_MPE_pct": float(np.mean(auc_err / g["auc_true"]) * 100),
        "auc_MAPE_pct": float(np.mean(np.abs(auc_err) / g["auc_true"]) * 100),
        "auc_PI90_coverage": float(np.mean((g["auc_true"] >= g["auc_pred_q05"]) & (g["auc_true"] <= g["auc_pred_q95"]))),
        "target_attainment_twin": float(np.mean(in_t(g["achieved_auc"]))),
        "target_attainment_standard": float(np.mean(in_t(g["standard_auc"]))),
        "overexposure_twin": float(np.mean(g["achieved_auc"] > target.auc_high)),
        "overexposure_standard": float(np.mean(g["standard_auc"] > target.auc_high)),
    }


def summarize(df: pd.DataFrame, target: TherapeuticTarget = TherapeuticTarget()) -> dict:
    return {m: _metrics(g, target) for m, g in df.groupby("method")}


def subgroup_table(df: pd.DataFrame, target: TherapeuticTarget = TherapeuticTarget()) -> pd.DataFrame:
    """Analisi di equita': prestazioni per sesso, eta' e funzione renale."""
    d = df.copy()
    d["eta"] = np.where(d["age"] >= 65, ">=65", "<65")
    d["funzione_renale"] = pd.cut(d["crcl"], [0, 60, 130, np.inf], labels=["CrCl<60", "60-130", ">130"]).astype(str)
    rows = []
    for var in ("sex", "eta", "funzione_renale"):
        for (level, method), g in d.groupby([var, "method"]):
            m = _metrics(g, target)
            rows.append({"variabile": var, "livello": level, "method": method, **{k: m[k] for k in ("n", "conc_MAPE_pct", "auc_MAPE_pct", "auc_PI90_coverage", "target_attainment_twin")}})
    return pd.DataFrame(rows)
