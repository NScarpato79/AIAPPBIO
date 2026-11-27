# Pneumonia Detection – MONAI Application Package

## Descrizione

Applicazione di Deep Learning per la classificazione binaria di immagini radiografiche toraciche, finalizzata al riconoscimento delle classi **Normal** e **Pneumonia**.

Il progetto integra due modelli:
- CNN personalizzata;
- Vision Transformer ViT-B/16.

L'applicazione è organizzata come MONAI Application Package (MAP) ed è eseguibile tramite container Docker. Supporta il caricamento di immagini DICOM locali e il recupero di istanze DICOM da un server PACS Orthanc.

## Dataset

Per l'addestramento e la valutazione dei modelli è stato utilizzato **PneumoniaMNIST**, benchmark appartenente alla raccolta MedMNIST e derivato da radiografie toraciche pediatriche.

Il dataset prevede due classi:
- `Normal`
- `Pneumonia`

Le immagini sono state usate nella versione 28 × 28 pixel e ridimensionate a 224 × 224 per i modelli.

## Modelli

- **CNN personalizzata**: rete convoluzionale progettata e addestrata sul dataset PneumoniaMNIST.
- **ViT-B/16**: Vision Transformer addestrato da zero (senza pesi preaddestrati) su PneumoniaMNIST.

Il modello predefinito è `vit`. È possibile selezionare il modello tramite il parametro `--model`.

## Architettura dell'applicazione

- `app.py`: punto di ingresso, gestione degli argomenti, acquisizione dell'immagine e orchestrazione dell'inferenza;
- `src/`: definizione dei modelli e logica di inferenza;
- `models/`: checkpoint addestrati della CNN e del ViT; --> CHECKPOINT NON INCLUSI NELLA REPO, SCARICABILI DA https://drive.google.com/drive/folders/1IeeCKbtQuF9cIW_ik3b2AMUPsQmOhMMo?usp=sharing. 
- `app.json`: manifest descrittivo dell'applicazione;
- `pkg.json`: manifest del package MAP;
- `Dockerfile`: configurazione per la creazione dell'immagine Docker.

## Input

L'applicazione supporta due modalità di acquisizione.

### Immagine DICOM locale

È possibile specificare il percorso di un file DICOM tramite `--input`.

In alternativa, è possibile montare una cartella in `/var/holoscan/input`. Se non viene specificato un file, l'applicazione individua automaticamente l'unico file `.dcm` presente nella cartella.

In modalità automatica deve essere presente un solo file DICOM.

### Recupero da PACS Orthanc

È possibile recuperare un'istanza DICOM direttamente da un server Orthanc tramite `--orthanc-instance`.

La connessione utilizza le seguenti variabili d'ambiente:

- `ORTHANC_URL`: indirizzo del server Orthanc;
- `ORTHANC_USERNAME`: nome utente;
- `ORTHANC_PASSWORD`: password;
- `ORTHANC_INSTANCE_ID`: identificativo dell'istanza DICOM predefinito.

## Esecuzione

### Build dell'immagine Docker

Dalla directory `deploy_app`:

    docker build -t pneumonia-detection-map:1.0.0 .

### Visualizzazione dei manifest

    docker run --rm pneumonia-detection-map:1.0.0 --show

### Estrazione dei manifest

    docker run --rm -v "$PWD/output:/export" pneumonia-detection-map:1.0.0 --extract /export/manifests

### Inferenza su un'immagine DICOM locale

    docker run --rm -v "$PWD/input:/var/holoscan/input" -v "$PWD/output:/var/holoscan/output" pneumonia-detection-map:1.0.0 --model vit

Per utilizzare la CNN, sostituire `vit` con `cnn`.

### Inferenza tramite Orthanc PACS

    docker run --rm -e ORTHANC_PASSWORD="password" -v "$PWD/output:/var/holoscan/output" pneumonia-detection-map:1.0.0 --orthanc-instance "INSTANCE_ID" --model vit

L'indirizzo e le credenziali del server possono essere configurati tramite le variabili d'ambiente corrispondenti.

## Output

Al termine dell'inferenza viene generato il file `prediction.json` nella directory `/var/holoscan/output`.

Il risultato contiene:
- classe predetta;
- probabilità associate alle classi;
- modello utilizzato;
- nome del file elaborato;
- sorgente dell'immagine (file locale o Orthanc PACS);
- dispositivo utilizzato per l'inferenza.

## Limitazioni tecniche

- L'attuale preprocessing è progettato per immagini DICOM 2D monocromatiche.
- In modalità automatica viene elaborato un singolo file DICOM per esecuzione.
- L'integrazione PACS è implementata tramite API REST di Orthanc.

Il package e' stato realizzato manualmente seguendo la specifica MAP, senza MONAI Deploy App SDK ne' il relativo Packager.
