"""Figure per relazione e dashboard (matplotlib, backend non interattivo)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import TherapeuticTarget  # noqa: E402
from .pk_model import concentration  # noqa: E402
from .scenarios import concentration_bands  # noqa: E402

COLORS = {"population_prior": "tab:gray", "bayes_pymc": "tab:blue", "enkf": "tab:orange"}
LABELS = {"population_prior": "A priori popolazione", "bayes_pymc": "Bayes (PyMC)", "enkf": "EnKF"}


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_patient_twin(patient, samples: dict, path: Path, t_max: float = 84.0):
    t = np.linspace(0, t_max, 600)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, s in samples.items():
        lo, med, hi = concentration_bands(s, patient.doses, t)
        ax.fill_between(t, lo, hi, color=COLORS[name], alpha=0.18)
        ax.plot(t, med, color=COLORS[name], label=f"{LABELS[name]} (mediana, banda 90%)")
    ax.plot(t, concentration(t, patient.doses, patient.cl_true, patient.v_true), "k--", lw=1, label="Paziente virtuale (verita')")
    fit = np.isfinite(patient.obs_conc[:2])
    ax.scatter(patient.obs_times[:2][fit], patient.obs_conc[:2][fit], c="red", zorder=5, label="Prelievi usati")
    ax.scatter(patient.obs_times[2], patient.obs_conc[2], marker="x", c="red", s=60, zorder=5, label="Prelievo di validazione")
    ax.set(xlabel="Tempo [h]", ylabel="Vancomicina [mg/L]", title=f"Digital Twin paziente {patient.pid}")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_enkf_history(history, cl_true: float, v_true: float, path: Path, t_end: float | None = None):
    history = list(history)
    if t_end is not None and t_end > history[-1].t:
        from dataclasses import replace

        history.append(replace(history[-1], t=t_end, y=None))
    t = [h.t for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, truth, unit in ((axes[0], "cl", cl_true, "CL [L/h]"), (axes[1], "v", v_true, "V [L]")):
        ax.fill_between(t, [getattr(h, f"{key}_q05") for h in history], [getattr(h, f"{key}_q95") for h in history], alpha=0.25, color="tab:orange", step="post")
        ax.step(t, [getattr(h, f"{key}_mean") for h in history], where="post", color="tab:orange", label="EnKF media (banda 90%)")
        ax.axhline(truth, color="k", ls="--", label="Valore vero")
        ax.set(xlabel="Tempo [h]", ylabel=unit)
        ax.legend(fontsize=8)
    fig.suptitle("Assimilazione sequenziale dei prelievi (EnKF)")
    _save(fig, path)


def plot_scenarios(table: pd.DataFrame, path: Path, target: TherapeuticTarget = TherapeuticTarget(), top: int = 12):
    t = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(t))
    ax.axvspan(target.auc_low, target.auc_high, color="green", alpha=0.12, label="Target AUC24 400-600")
    ax.errorbar(t["auc_median"], y, xerr=[t["auc_median"] - t["auc_q05"], t["auc_q95"] - t["auc_median"]], fmt="o", capsize=3)
    ax.set_yticks(y, [f"{r}  (PTA {p:.0%})" for r, p in zip(t["regimen"], t["pta"])])
    ax.set(xlabel="AUC24 predetta [mg*h/L] (mediana, IC 90%)", title="Scenari terapeutici simulati dal Digital Twin")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_scenario_bands(samples, doses_standard, doses_twin, t_dec: float, path: Path, labels=("Regime standard", "Regime raccomandato")):
    t = np.linspace(0, t_dec + 48, 600)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for doses, label, color in ((doses_standard, labels[0], "tab:red"), (doses_twin, labels[1], "tab:green")):
        lo, med, hi = concentration_bands(samples, doses, t)
        ax.fill_between(t, lo, hi, alpha=0.2, color=color)
        ax.plot(t, med, color=color, label=f"{label} (mediana, banda 90%)")
    ax.axvline(t_dec, color="k", ls=":", label="Decisione terapeutica")
    ax.set(xlabel="Tempo [h]", ylabel="Vancomicina [mg/L]", title="Confronto scenari con bande di incertezza predittiva")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_calibration(df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for name, g in df.groupby("method"):
        axes[0].scatter(g["obs_holdout"], g["pred_median"], s=14, alpha=0.7, color=COLORS[name], label=LABELS[name])
        axes[1].scatter(g["auc_true"], g["auc_pred_median"], s=14, alpha=0.7, color=COLORS[name], label=LABELS[name])
    for ax, lab in ((axes[0], "Valle a 72 h [mg/L]"), (axes[1], "AUC24 48-72 h [mg*h/L]")):
        lim = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set(xlabel=f"Osservato/vero: {lab}", ylabel="Predetto (mediana)")
        ax.legend(fontsize=8)
    fig.suptitle("Validazione interna sul prelievo tenuto da parte")
    _save(fig, path)


def plot_attainment(summary: dict, path: Path):
    methods = list(summary)
    std = summary[methods[0]]["target_attainment_standard"]
    vals = [std] + [summary[m]["target_attainment_twin"] for m in methods]
    names = ["Regime standard"] + [f"Twin: {LABELS[m]}" for m in methods]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, vals, color=["tab:red"] + [COLORS[m] for m in methods])
    ax.bar_label(bars, labels=[f"{v:.0%}" for v in vals])
    ax.set(ylim=(0, 1), ylabel="Pazienti con AUC24 vera nel target", title="Raggiungimento del target terapeutico")
    plt.setp(ax.get_xticklabels(), rotation=15)
    _save(fig, path)


def plot_sobol(df: pd.DataFrame, path: Path):
    outputs = df["output"].unique()
    fig, axes = plt.subplots(1, len(outputs), figsize=(10, 3.8))
    for ax, out in zip(np.atleast_1d(axes), outputs):
        g = df[df["output"] == out]
        x = np.arange(len(g))
        ax.bar(x - 0.2, g["S1"].clip(lower=0), 0.4, label="S1 (primo ordine)")
        ax.bar(x + 0.2, g["ST"].clip(lower=0), 0.4, label="ST (totale)")
        ax.set_xticks(x, g["factor"])
        ax.set(title=f"Sensitivita' globale: {out}", ylim=(0, 1))
        ax.legend(fontsize=8)
    _save(fig, path)
