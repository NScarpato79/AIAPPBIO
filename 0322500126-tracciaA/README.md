# VancoTwin — Digital Twin PK/PD della vancomicina (Traccia A)

Prototipo di **Digital Twin fisiologico con assimilazione dati in tempo reale** per la
personalizzazione della terapia con vancomicina (target AUC24/MIC 400–600 mg·h/L).

| Requisito della traccia | Implementazione |
|---|---|
| 1. Dinamica temporale con ODE | `pk_model.py` — ODE monocompartimentale con infusione intermittente (scipy `solve_ivp`, vettorizzata sull'ensemble) + soluzione analitica esatta verificata dai test |
| 2. Calibrazione bayesiana con PyMC | `calibration.py` — a priori di popolazione con covariate (CrCl, peso), parametrizzazione non centrata, NUTS, diagnostiche R̂/ESS/divergenze |
| 3. Filtro bayesiano real-time (EnKF) | `enkf.py` — stato aumentato `[C, log CL, log V]`, forecast via ODE, analysis con osservazioni perturbate e inflazione |
| 4. Scenari terapeutici con incertezza | `scenarios.py` — regimi candidati (dose × intervallo), distribuzione predittiva di AUC24, PTA, rischio di tossicità, bande al 90% |

Moduli trasversali: `data.py` (coorte virtuale, QC, pseudonimizzazione HMAC, minimizzazione),
`evaluation.py` (validazione su prelievo tenuto da parte, analisi per sottogruppi),
`sensitivity.py` (spiegabilità: indici di Sobol, shrinkage del posterior), `plotting.py`, `pipeline.py`.

## Struttura

```
src/vancotwin/      codice del pacchetto
tests/              test unitari (pytest), incluso il test "slow" di PyMC
outputs/            risultati generati: metrics.json, CSV, figures/
Dockerfile          ambiente riproducibile (Python 3.11)
environment.yml     ambiente conda alternativo
```

## Esecuzione

### Con Docker 

```bash
docker build -t vancotwin .
docker run --rm vancotwin pytest -q                                                  # test unitari
docker run --rm -v "$PWD/outputs:/app/outputs" vancotwin                             # pipeline completa (~10 min CPU)
docker run --rm -v "$PWD/outputs:/app/outputs" vancotwin python -m vancotwin --quick # demo ~1 min
```

### Con conda

```bash
conda env create -f environment.yml
conda activate vancotwin
pytest -q
python -m vancotwin --output outputs
```

Opzioni: `--quick` (coorte ridotta, meno campioni MCMC), `--n-eval N`, `--seed S`.
Il segreto per la pseudonimizzazione si imposta con la variabile d'ambiente `VANCOTWIN_SALT`.

## Pipeline

1. **Ingestione**: coorte virtuale di 200 pazienti adulti (età, sesso, peso, creatinina → CrCl Cockcroft-Gault),
   parametri veri CL/V con variabilità inter-individuale correlata, regime empirico (carico 25 mg/kg + 15 mg/kg q12h),
   3 prelievi TDM con errore combinato e anomalie simulate (mancanti, errori di unità di misura).
2. **Quality control**: mancanti, negativi, valori implausibili (>150 mg/L), duplicati → esclusione.
3. **Personalizzazione**: i primi due prelievi (picco ~14 h, valle ~35,5 h) calibrano il twin con PyMC e con l'EnKF.
4. **Decisione a 36 h**: il twin simula i regimi candidati e raccomanda quello con massima PTA.
5. **Validazione**: predizione del prelievo tenuto da parte (valle ~71,5 h) e dell'AUC vera; AUC ottenuta
   con il regime raccomandato vs regime standard; metriche per sesso, età e funzione renale.
6. **Spiegabilità**: indici di Sobol di AUC24 e Cvalle; riduzione dell'incertezza (shrinkage).

## Output principali

- `outputs/metrics.json`: QC, caso dimostrativo, metriche di validazione, diagnostiche MCMC, Sobol
- `outputs/evaluation_per_patient.csv`, `evaluation_subgroups.csv`, `demo_scenarios.csv`, `sobol_indices.csv`
- `outputs/figures/01…07_*.png`
