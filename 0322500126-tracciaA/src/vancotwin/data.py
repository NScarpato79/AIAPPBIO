"""Generazione della coorte virtuale, quality control e pseudonimizzazione.

I pazienti sono *sintetici*: i parametri "veri" (CL, V) sono estratti dal modello di
popolazione con variabilita' inter-individuale. Questo permette di conoscere la
verita' di riferimento (ground truth) e quindi di validare il Digital Twin in modo
rigoroso, senza trattare dati personali reali (GDPR by design).
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import PopulationModel
from .pk_model import Dose, concentration, infusion_duration, regimen

PLAUSIBLE_MAX_CONC = 150.0  # mg/L: oltre e' fisiologicamente implausibile (errore di unita'/trascrizione)
DEFAULT_SALT = "vancotwin-demo-salt"


def cockcroft_gault(age, weight, scr, female) -> np.ndarray:
    """Clearance della creatinina stimata [mL/min] (formula di Cockcroft-Gault)."""
    age, weight, scr = (np.asarray(x, dtype=float) for x in (age, weight, scr))
    crcl = (140.0 - age) * weight / (72.0 * scr)
    return np.where(np.asarray(female, dtype=bool), 0.85 * crcl, crcl)


def round_dose(mg: float, step: float = 250.0, max_mg: float | None = None) -> float:
    d = max(step, step * round(mg / step))
    return min(d, max_mg) if max_mg else d


def standard_initial_regimen(weight: float, horizon_h: float = 96.0) -> list[Dose]:
    """Dosaggio empirico: carico 25 mg/kg (max 3 g) + mantenimento 15 mg/kg ogni 12 h."""
    loading = round_dose(25.0 * weight, max_mg=3000.0)
    maint = round_dose(15.0 * weight, max_mg=2000.0)
    doses = [Dose(0.0, loading, infusion_duration(loading))]
    doses += regimen(maint, 12.0, int(horizon_h // 12), start=12.0)
    return doses


def pseudonymize(identifier: str, salt: str | None = None) -> str:
    """Pseudonimo deterministico (HMAC-SHA256). Senza il segreto non e' reversibile."""
    key = (salt or os.environ.get("VANCOTWIN_SALT") or DEFAULT_SALT).encode()
    return "P-" + hmac.new(key, identifier.encode(), hashlib.sha256).hexdigest()[:12]


def age_band(age: float) -> str:
    lo = int(age // 10 * 10)
    return f"{lo}-{lo + 9}"


@dataclass
class Patient:
    pid: str
    age: float
    female: bool
    weight: float
    scr: float
    crcl: float
    cl_true: float
    v_true: float
    doses: list[Dose]
    obs_times: np.ndarray
    obs_conc: np.ndarray
    holdout_index: int = 2
    meta: dict = field(default_factory=dict)

    @property
    def sex(self) -> str:
        return "F" if self.female else "M"


def _observation_times(doses: list[Dose], rng: np.random.Generator) -> np.ndarray:
    """Campioni TDM: picco dopo la 1a dose di mantenimento, valle prima delle dosi a 36 h e 72 h."""
    first_maint = doses[1]
    t_peak = first_maint.end + 1.0 + rng.uniform(-0.25, 0.25)
    t_trough1 = 36.0 - 0.5 + rng.uniform(-0.25, 0.25)
    t_trough2 = 72.0 - 0.5 + rng.uniform(-0.25, 0.25)
    return np.array([t_peak, t_trough1, t_trough2])


def generate_cohort(
    n: int,
    rng: np.random.Generator,
    pop: PopulationModel = PopulationModel(),
    missing_rate: float = 0.02,
    gross_error_rate: float = 0.01,
    salt: str | None = None,
) -> list[Patient]:
    patients = []
    for i in range(n):
        age = float(rng.uniform(18, 90))
        female = bool(rng.random() < 0.45)
        weight = float(np.clip(rng.lognormal(np.log(65 if female else 78), 0.18), 40, 150))
        scr = float(np.clip(rng.lognormal(np.log(0.9) + 0.006 * (age - 50), 0.35), 0.4, 5.0))
        crcl = float(cockcroft_gault(age, weight, scr, female))
        eta = rng.multivariate_normal(
            [0.0, 0.0],
            [[pop.omega_cl**2, 0.3 * pop.omega_cl * pop.omega_v],
             [0.3 * pop.omega_cl * pop.omega_v, pop.omega_v**2]],
        )
        cl_true = float(pop.cl_pop(crcl) * np.exp(eta[0]))
        v_true = float(pop.v_pop(weight) * np.exp(eta[1]))
        doses = standard_initial_regimen(weight)
        t_obs = _observation_times(doses, rng)
        c_true = concentration(t_obs, doses, cl_true, v_true)
        c_obs = np.maximum(c_true + rng.normal(0.0, pop.obs_sd(c_true)), 0.1)
        # Anomalie realistiche da intercettare nel quality control
        for j in range(2):
            u = rng.random()
            if u < missing_rate:
                c_obs[j] = np.nan
            elif u < missing_rate + gross_error_rate:
                c_obs[j] *= 1000.0  # errore di unita' di misura (ug/L invece di mg/L)
        mrn = f"MRN{100000 + i}"
        patients.append(
            Patient(
                pid=pseudonymize(mrn, salt),
                age=age,
                female=female,
                weight=weight,
                scr=scr,
                crcl=crcl,
                cl_true=cl_true,
                v_true=v_true,
                doses=doses,
                obs_times=t_obs,
                obs_conc=c_obs,
            )
        )
    return patients


def cohort_tables(patients: list[Patient]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabelle pseudonimizzate e minimizzate (fascia d'eta' invece dell'eta' esatta)."""
    demo = pd.DataFrame(
        {
            "pid": p.pid,
            "age_band": age_band(p.age),
            "sex": p.sex,
            "weight_kg": round(p.weight, 1),
            "scr_mg_dl": round(p.scr, 2),
            "crcl_ml_min": round(p.crcl, 1),
        }
        for p in patients
    )
    obs = pd.DataFrame(
        {"pid": p.pid, "sample": j, "time_h": round(float(t), 2), "conc_mg_l": c, "holdout": j == p.holdout_index}
        for p in patients
        for j, (t, c) in enumerate(zip(p.obs_times, p.obs_conc))
    )
    return demo, obs


def quality_control(obs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Segnala e rimuove misure mancanti, negative, implausibili o duplicate."""
    df = obs.copy()
    df["qc_missing"] = df["conc_mg_l"].isna()
    df["qc_negative"] = df["conc_mg_l"] < 0
    df["qc_implausible"] = df["conc_mg_l"] > PLAUSIBLE_MAX_CONC
    df["qc_duplicate"] = df.duplicated(subset=["pid", "time_h"], keep="first")
    flags = ["qc_missing", "qc_negative", "qc_implausible", "qc_duplicate"]
    df["qc_pass"] = ~df[flags].any(axis=1)
    report = {f: int(df[f].sum()) for f in flags}
    report.update(n_total=int(len(df)), n_pass=int(df["qc_pass"].sum()))
    return df, report


def apply_qc(patients: list[Patient], qc_df: pd.DataFrame) -> None:
    """Imposta a NaN le misure scartate dal QC (verranno ignorate da calibrazione ed EnKF)."""
    bad = qc_df.loc[~qc_df["qc_pass"], ["pid", "sample"]]
    by_pid = {p.pid: p for p in patients}
    for pid, j in bad.itertuples(index=False):
        by_pid[pid].obs_conc[j] = np.nan
