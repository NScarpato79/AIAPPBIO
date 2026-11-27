import torch
import numpy as np

from pathlib import Path
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    ScaleIntensity,
    Resize,
    NormalizeIntensity,
    EnsureType,
)

from .model_definitions import PneumoniaCNN, create_vit


CLASS_NAMES = ["Normal", "Pneumonia"]
IMAGE_SIZE = 224


def get_transform(model_name):
    transforms = [
        EnsureChannelFirst(channel_dim=-1),
        ScaleIntensity(),
        Resize(spatial_size=(IMAGE_SIZE, IMAGE_SIZE)),
    ]

    if model_name == "vit":
        transforms.append(
            NormalizeIntensity(
                subtrahend=[0.485, 0.456, 0.406],
                divisor=[0.229, 0.224, 0.225],
                channel_wise=True,
            )
        )

    transforms.append(EnsureType(dtype=torch.float32))
    return Compose(transforms)


def load_model(model_name, project_dir=None, device=None, models_dir=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if models_dir is not None:
        models_path = Path(models_dir)
    elif project_dir is not None:
        models_path = Path(project_dir) / "models"
    else:
        models_path = Path(__file__).resolve().parents[1] / "models"

    if model_name == "cnn":
        model = PneumoniaCNN()
        weights_path = models_path / "best_pneumonia_cnn.pth"
    elif model_name == "vit":
        model = create_vit()
        weights_path = models_path / "best_pneumonia_vit.pth"
    else:
        raise ValueError("Modello non supportato. Usa 'cnn' oppure 'vit'.")

    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint non trovato: {weights_path}")

    state_dict = torch.load(
        weights_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, device


def predict(image, model, model_name, device):
    image_array = np.array(image, copy=True)
    transform = get_transform(model_name)
    image_tensor = transform(image_array).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    predicted_class = int(torch.argmax(probabilities, dim=1).item())

    return {
        "predicted_class": CLASS_NAMES[predicted_class],
        "probabilities": {
            CLASS_NAMES[i]: float(probabilities[0, i].item())
            for i in range(len(CLASS_NAMES))
        },
    }
