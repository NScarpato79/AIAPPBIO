"""Test di modelli, preprocessing, lettura DICOM e applicazione.

I test sui pesi addestrati sono saltati se i file .pth non sono presenti.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

DEPLOY = Path(__file__).resolve().parents[1] / "deploy_app"
sys.path.insert(0, str(DEPLOY))

import app  # noqa: E402
from src.inference import get_transform, load_model, predict  # noqa: E402
from src.model_definitions import PneumoniaCNN  # noqa: E402

SAMPLE_DICOM = DEPLOY / "input" / "test_xray_from_pacs.dcm"
IMAGE = np.random.default_rng(0).integers(0, 256, (28, 28, 3), dtype=np.uint8)


class TestPreprocessingAndModel(unittest.TestCase):
    def test_transform_shape(self):
        for name in ("cnn", "vit"):
            self.assertEqual(tuple(get_transform(name)(IMAGE).shape), (3, 224, 224))

    def test_prediction_is_a_distribution(self):
        model = PneumoniaCNN().eval()
        result = predict(IMAGE, model, "cnn", torch.device("cpu"))
        self.assertIn(result["predicted_class"], ["Normal", "Pneumonia"])
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)

    def test_unsupported_model(self):
        with self.assertRaises(ValueError):
            load_model("unsupported")

    def test_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError):
                load_model("cnn", models_dir=empty)


class TestApplication(unittest.TestCase):
    def test_dicom_is_read_as_rgb(self):
        image = app.load_dicom_image(SAMPLE_DICOM)
        self.assertEqual(image.mode, "RGB")

    def test_orthanc_requires_password(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                app.download_from_orthanc("id", Path("unused.dcm"))

    def test_orthanc_download(self):
        response = mock.Mock(content=b"dicom-bytes")
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.dict(os.environ, {"ORTHANC_PASSWORD": "x"}), \
                mock.patch.object(app.requests, "get", return_value=response) as get:
            path = app.download_from_orthanc("abc", Path(folder) / "a.dcm")
            self.assertEqual(path.read_bytes(), b"dicom-bytes")
            self.assertTrue(get.call_args.args[0].endswith("/instances/abc/file"))


@unittest.skipUnless((DEPLOY / "models" / "best_pneumonia_cnn.pth").is_file(),
                     "pesi non presenti")
class TestEndToEnd(unittest.TestCase):
    def test_dicom_to_json(self):
        with tempfile.TemporaryDirectory() as out, \
                mock.patch.dict(os.environ, {"HOLOSCAN_MODEL_PATH": str(DEPLOY / "models")}), \
                mock.patch.object(sys, "argv", ["app.py", "--input", str(SAMPLE_DICOM),
                                                "--model", "cnn", "--output", out]):
            os.environ.pop("ORTHANC_INSTANCE_ID", None)
            app.main()
            result = json.loads((Path(out) / "prediction.json").read_text(encoding="utf-8"))
            self.assertEqual(result["source"], "local file")


if __name__ == "__main__":
    unittest.main()
