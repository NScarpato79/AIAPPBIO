import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pydicom
import requests
from PIL import Image

from src.inference import load_model, predict


def download_from_orthanc(instance_id: str, destination: Path) -> Path:
    """Scarica un'istanza DICOM dal PACS Orthanc tramite REST API."""
    base_url = os.getenv(
        "ORTHANC_URL",
        "http://host.docker.internal:8042"
    ).rstrip("/")

    username = os.getenv("ORTHANC_USERNAME", "admin")
    password = os.getenv("ORTHANC_PASSWORD")

    if not password:
        raise ValueError(
            "Configurare la variabile d'ambiente ORTHANC_PASSWORD."
        )

    url = f"{base_url}/instances/{instance_id}/file"

    response = requests.get(
        url,
        auth=(username, password),
        timeout=60,
    )
    response.raise_for_status()

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)

    return destination


def load_dicom_image(dicom_path: Path) -> Image.Image:
    """Legge un DICOM e restituisce un'immagine RGB compatibile con il modello."""
    dataset = pydicom.dcmread(str(dicom_path))

    if not hasattr(dataset, "PixelData"):
        raise ValueError("Il file DICOM non contiene dati pixel.")

    pixels = dataset.pixel_array.astype(np.float32)

    if pixels.ndim != 2:
        raise ValueError(
            f"Immagine DICOM non supportata: attese 2 dimensioni, "
            f"ricevute {pixels.ndim}."
        )

    slope = float(getattr(dataset, "RescaleSlope", 1))
    intercept = float(getattr(dataset, "RescaleIntercept", 0))
    pixels = pixels * slope + intercept

    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = pixels.max() + pixels.min() - pixels

    pixels -= pixels.min()
    maximum = pixels.max()

    if maximum > 0:
        pixels = pixels / maximum * 255.0

    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    rgb = np.repeat(pixels[..., np.newaxis], 3, axis=-1)

    return Image.fromarray(rgb)


def main():
    parser = argparse.ArgumentParser(
        description="Pneumonia Detection - MAP"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Percorso del file DICOM locale."
    )
    parser.add_argument(
        "--orthanc-instance",
        default=os.getenv("ORTHANC_INSTANCE_ID"),
        help="ID dell'istanza DICOM da recuperare da Orthanc."
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "vit"],
        default=os.getenv("PNEUMONIA_MODEL", "vit"),
        help="Modello da utilizzare (default: vit)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.getenv("HOLOSCAN_OUTPUT_PATH", "/output")),
        help="Cartella in cui salvare il risultato JSON."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostra i manifest MAP installati nel container."
    )

    parser.add_argument(
        "--extract",
        type=Path,
        default=None,
        help="Cartella in cui copiare i manifest MAP."
    )

    args = parser.parse_args()

    if args.show:
        for manifest in (
            Path("/etc/holoscan/app.json"),
            Path("/etc/holoscan/pkg.json"),
        ):
            print(f"\n--- {manifest} ---")
            if manifest.is_file():
                print(manifest.read_text(encoding="utf-8"))
            else:
                print("Manifest non ancora disponibile.")
        return

    if args.extract is not None:
        args.extract.mkdir(parents=True, exist_ok=True)
        for manifest_name in ("app.json", "pkg.json"):
            source = Path("/etc/holoscan") / manifest_name
            if not source.is_file():
                raise FileNotFoundError(f"Manifest non trovato: {source}")
            shutil.copy2(source, args.extract / manifest_name)
        print(f"Manifest copiati in: {args.extract.resolve()}")
        return

    if args.input is not None and args.orthanc_instance is not None:
        parser.error("Specificare --input oppure --orthanc-instance, non entrambi.")

    if args.orthanc_instance:
        dicom_path = download_from_orthanc(
            args.orthanc_instance,
            Path("/tmp/pneumonia-map-input/input.dcm"),
        )
    elif args.input is not None:
        dicom_path = args.input
    else:
        input_dir = Path(os.getenv("HOLOSCAN_INPUT_PATH", "/var/holoscan/input"))
        dicom_files = sorted(input_dir.glob("*.dcm"))

        if len(dicom_files) == 1:
            dicom_path = dicom_files[0]
        elif len(dicom_files) == 0:
            parser.error(
                f"Nessun DICOM trovato in {input_dir}. "
                "Montare un file .dcm oppure specificare --input "
                "o ORTHANC_INSTANCE_ID."
            )
        else:
            parser.error(
                f"Trovati {len(dicom_files)} file DICOM in {input_dir}. "
                "Specificare il file desiderato con --input."
            )

    if not dicom_path.is_file():
        raise FileNotFoundError(f"File DICOM non trovato: {dicom_path}")

    model_dir = Path(
        os.getenv("HOLOSCAN_MODEL_PATH", "/opt/holoscan/models")
    )

    model, device = load_model(
        model_name=args.model,
        device=None,
        models_dir=model_dir,
    )

    image = load_dicom_image(dicom_path)
    result = predict(image, model, args.model, device)

    result.update({
        "model": args.model,
        "input_file": dicom_path.name,
        "source": "Orthanc PACS" if args.orthanc_instance else "local file",
        "device": str(device),
        "disclaimer": (
            "Risultato sperimentale su dati a scopo didattico; "
            "non utilizzabile per diagnosi cliniche."
        ),
    })

    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / "prediction.json"

    output_file.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nRisultato salvato in: {output_file}")


if __name__ == "__main__":
    main()
