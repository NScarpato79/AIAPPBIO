"""Orchestrazione end-to-end della pipeline del Digital Twin."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import plotting
from .config import PopulationModel, RunConfig, TherapeuticTarget
from .data import apply_qc, cohort_tables, generate_cohort, quality_control
from .evaluation import evaluate_patient, personalize, subgroup_table, summarize
from .scenarios import apply_regimen, recommend
from .sensitivity import population_sensitivity, posterior_shrinkage

log = logging.getLogger("vancotwin")


def _demo_patient(patients, pop: PopulationModel, target: TherapeuticTarget):
    """Sceglie un paziente con entrambi i prelievi validi e AUC standard fuori target (caso didattico)."""
    for p in patients:
        if np.isfinite(p.obs_conc[:2]).all():
            auc_std = 2 * p.doses[1].amount / p.cl_true
            if auc_std > target.auc_high:
                return p
    return patients[0]


def run(cfg: RunConfig = RunConfig(), pop: PopulationModel = PopulationModel(), target: TherapeuticTarget = TherapeuticTarget()) -> dict:
    t_start = time.time()
    out = Path(cfg.output_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    # 1. Ingestione: coorte virtuale pseudonimizzata
    log.info("1/6 Generazione coorte virtuale (%d pazienti)", cfg.n_patients)
    patients = generate_cohort(cfg.n_patients, rng, pop)
    demo, obs = cohort_tables(patients)

    # 2. Quality control
    log.info("2/6 Quality control")
    qc_df, qc_report = quality_control(obs)
    apply_qc(patients, qc_df)
    demo.to_csv(out / "cohort_demographics.csv", index=False)
    qc_df.to_csv(out / "tdm_observations_qc.csv", index=False)

    # 3. Caso dimostrativo: calibrazione PyMC + EnKF + scenari
    log.info("3/6 Digital Twin del paziente dimostrativo")
    p = _demo_patient(patients, pop, target)
    prior, bayes, enkf = personalize(p, cfg, pop, rng, cfg.seed)
    twins = {"population_prior": prior, "bayes_pymc": bayes, "enkf": enkf.samples()}
    plotting.plot_patient_twin(p, twins, fig_dir / "01_twin_paziente.png")
    plotting.plot_enkf_history(enkf.history, p.cl_true, p.v_true, fig_dir / "02_enkf_assimilazione.png", t_end=cfg.decision_time + 12.0)
    best, scen = recommend(bayes, p.doses, cfg.decision_time, target, rng=rng)
    scen.to_csv(out / "demo_scenarios.csv", index=False)
    plotting.plot_scenarios(scen, fig_dir / "03_scenari_auc.png", target)
    plotting.plot_scenario_bands(bayes, p.doses, apply_regimen(p.doses, best, cfg.decision_time), cfg.decision_time, fig_dir / "04_scenari_bande.png")
    demo_summary = {
        "pid": p.pid,
        "crcl": p.crcl,
        "weight": p.weight,
        "true": {"CL": p.cl_true, "V": p.v_true},
        "prior": prior.summary(),
        "bayes_pymc": bayes.summary(),
        "enkf": enkf.samples().summary(),
        "shrinkage_CL_bayes": posterior_shrinkage(np.std(np.log(prior.cl)), np.std(np.log(bayes.cl))),
        "shrinkage_CL_enkf": posterior_shrinkage(np.std(np.log(prior.cl)), np.std(enkf.log_cl)),
        "recommended": best.label,
        "scenario_top5": scen.head(5).to_dict(orient="records"),
    }

    # 4. Validazione sulla coorte
    log.info("4/6 Validazione su %d pazienti", cfg.n_eval)
    rows = []
    for i, pt in enumerate(patients[: cfg.n_eval]):
        rows += evaluate_patient(pt, cfg, pop, target, rng, cfg.seed + i)
        log.info("   paziente %d/%d", i + 1, cfg.n_eval)
    ev = pd.DataFrame(rows)
    ev.to_csv(out / "evaluation_per_patient.csv", index=False)
    summary = summarize(ev, target)
    sub = subgroup_table(ev, target)
    sub.to_csv(out / "evaluation_subgroups.csv", index=False)
    plotting.plot_calibration(ev, fig_dir / "05_validazione.png")
    plotting.plot_attainment(summary, fig_dir / "06_target_attainment.png")

    # 5. Spiegabilita': sensitivita' globale
    log.info("5/6 Analisi di sensitivita' (Sobol)")
    sobol = population_sensitivity(8192, rng, pop)
    sobol.to_csv(out / "sobol_indices.csv", index=False)
    plotting.plot_sobol(sobol, fig_dir / "07_sobol.png")

    # 6. Report delle metriche
    log.info("6/6 Salvataggio metriche")
    bayes_rows = ev[ev["method"] == "bayes_pymc"]
    metrics = {
        "config": asdict(cfg),
        "population_model": asdict(pop),
        "target": asdict(target),
        "qc": qc_report,
        "demo_patient": demo_summary,
        "validation": summary,
        "mcmc": {
            "r_hat_max": float(bayes_rows["r_hat_max"].max()),
            "divergences_total": int(bayes_rows["divergences"].sum()),
        },
        "sobol": sobol.to_dict(orient="records"),
        "runtime_s": round(time.time() - t_start, 1),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log.info("Completato in %.0f s -> %s", metrics["runtime_s"], out.resolve())
    return metrics
