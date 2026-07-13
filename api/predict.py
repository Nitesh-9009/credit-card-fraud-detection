

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import numpy as np

_ARTIFACT = Path(__file__).resolve().parent / "model.npz"

# Load once per warm container.
_data = np.load(_ARTIFACT, allow_pickle=True)
_X_TRAIN = _data["X_train"].astype(np.float64)
_Y_TRAIN = _data["y_train"].astype(np.int64)
_MEAN = _data["mean"].astype(np.float64)
_SCALE = _data["scale"].astype(np.float64)
_K = int(_data["k"])
_FEATURES = [str(f) for f in _data["features"].tolist()]


def _vectorize(payload: dict) -> np.ndarray:
    if "features" in payload:
        values = payload["features"]
        if len(values) != len(_FEATURES):
            raise ValueError(
                f"Expected {len(_FEATURES)} features, got {len(values)}."
            )
        return np.asarray(values, dtype=np.float64)
    try:
        return np.asarray([float(payload[name]) for name in _FEATURES], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"Missing feature: {exc.args[0]}") from exc


def predict(payload: dict) -> dict:
    x = _vectorize(payload)
    x_scaled = (x - _MEAN) / _SCALE

    distances = np.linalg.norm(_X_TRAIN - x_scaled, axis=1)
    nearest = np.argpartition(distances, _K)[:_K]
    neighbor_labels = _Y_TRAIN[nearest]

    fraud_votes = int(neighbor_labels.sum())
    fraud_probability = fraud_votes / _K
    is_fraud = fraud_probability >= 0.5

    return {
        "prediction": int(is_fraud),
        "label": "fraud" if is_fraud else "valid",
        "fraud_probability": round(float(fraud_probability), 4),
        "k": _K,
        "fraud_votes": fraud_votes,
    }


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:  # CORS preflight
        self._send(204, {})

    def do_GET(self) -> None:
        self._send(200, {"status": "ok", "model": "knn", "k": _K, "features": _FEATURES})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw or b"{}")
            self._send(200, predict(payload))
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": f"Internal error: {exc}"})
