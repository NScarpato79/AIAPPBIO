# Traccia D - Classificazione della polmonite e deployment (PneumoniaMNIST)

Prototipo didattico: **non utilizzabile per diagnosi cliniche**.

## Risultati sul test set (624 immagini)

| Modello | ROC-AUC | Sensibilita' | Specificita' | Brier |
| --- | --- | --- | --- | --- |
| CNN personalizzata | 0,920 | 0,915 | 0,735 | 0,110 |
| ViT-B/16 (da zero) | 0,955 | 0,969 | 0,752 | 0,089 |

## Struttura

- `deploy_app/` applicazione (MAP), modelli e inferenza in `deploy_app/src/`
- `training/train.py` addestramento
- `scripts/evaluate.py` metriche, calibrazione, Grad-CAM
- `notebooks/` controllo qualita', valutazione, LIME
- `tests/` test unitari
- `results/` risultati
- `docs/` relazione

## Pesi

I file `.pth` non sono nel repository (il ViT supera i 100 MB).
Copiare `best_pneumonia_cnn.pth` e `best_pneumonia_vit.pth` in `deploy_app/models/`.

I pesi usati nei risultati sono stati addestrati in una precedente versione del
notebook (Google Colab, GPU). Il notebook consegnato li ricarica e non riaddestra
(`RIADDESTRA = False`). I valori per epoca di quell'addestramento non sono stati conservati.

## Esecuzione

Ambiente (Python 3.11):

    docker build -t pneumonia-dev .
    docker run --rm pneumonia-dev                                   # test
    docker run --rm -v "$PWD:/workspace" pneumonia-dev python scripts/evaluate.py
    docker run --rm -v "$PWD:/workspace" pneumonia-dev python training/train.py --model cnn

Applicazione:

    docker build -t pneumonia-detection-map:1.0.0 deploy_app

Per input DICOM e Orthanc vedere `deploy_app/README.md`.

## Dati

PneumoniaMNIST (MedMNIST), split ufficiali. Il controllo qualita' ha trovato 8 immagini
identiche tra training e validation e nessuna tra test e gli altri split.
