"""Parametri del modello di popolazione e della pipeline.

I valori tipici sono arrotondati e ispirati a modelli di popolazione pubblicati per
la vancomicina in adulti (clearance proporzionale alla funzione renale, volume
proporzionale al peso). Sono usati a scopo didattico, non per uso clinico.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PopulationModel:
    """Modello monocompartimentale con covariate.

    CL = cl_typ * (CrCl / 100) ** cl_crcl_exp      [L/h]
    V  = v_typ_per_kg * peso                        [L]
    Variabilita' inter-individuale log-normale (omega) ed errore residuo combinato
    additivo + proporzionale (sigma_add, sigma_prop).
    """

    cl_typ: float = 4.0
    cl_crcl_exp: float = 0.75
    crcl_cap: float = 150.0
    v_typ_per_kg: float = 0.7
    omega_cl: float = 0.35
    omega_v: float = 0.25
    sigma_prop: float = 0.15
    sigma_add: float = 1.0

    def cl_pop(self, crcl):
        import numpy as np

        crcl = np.minimum(np.asarray(crcl, dtype=float), self.crcl_cap)
        return self.cl_typ * (crcl / 100.0) ** self.cl_crcl_exp

    def v_pop(self, weight):
        import numpy as np

        return self.v_typ_per_kg * np.asarray(weight, dtype=float)

    def obs_sd(self, conc):
        import numpy as np

        conc = np.asarray(conc, dtype=float)
        return np.sqrt(self.sigma_add**2 + (self.sigma_prop * conc) ** 2)


@dataclass(frozen=True)
class TherapeuticTarget:
    """Target AUC24/MIC raccomandato per la vancomicina (MIC = 1 mg/L)."""

    auc_low: float = 400.0
    auc_high: float = 600.0
    mic: float = 1.0
    trough_toxic: float = 20.0


@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    n_patients: int = 200
    n_eval: int = 60
    draws: int = 1000
    tune: int = 1000
    chains: int = 2
    n_ensemble: int = 300
    decision_time: float = 36.0
    output_dir: str = "outputs"
    pseudonym_salt_env: str = "VANCOTWIN_SALT"
