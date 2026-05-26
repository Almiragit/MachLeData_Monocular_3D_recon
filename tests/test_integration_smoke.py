"""
tests/test_integration_smoke.py
--------------------------------
Smoke/integration tests for critical pipeline paths:
- Retrain trigger logic (Prometheus query + DVC trigger)
- FastAPI health + predict/json endpoint contract (with mocked model)
"""

import io
import sys
import types

import numpy as np
from PIL import Image


def test_query_prometheus_parses_value(monkeypatch):
    from src.training import retrain_trigger as rt

    class _Resp:
        def json(self):
            return {
                "data": {
                    "result": [{"value": [0, "1"]}]
                }
            }

    monkeypatch.setattr(rt.requests, "get", lambda *args, **kwargs: _Resp())
    value = rt.query_prometheus("drift_alert_triggered")
    assert value == 1.0


def test_trigger_retraining_success(monkeypatch):
    from src.training import retrain_trigger as rt

    class _Result:
        returncode = 0

    monkeypatch.setattr(rt.subprocess, "run", lambda *args, **kwargs: _Result())
    assert rt.trigger_retraining() is True


def test_api_health_endpoint(monkeypatch):
    class _DummyInstrumentator:
        def instrument(self, app):
            return self

        def expose(self, app):
            return self

    monkeypatch.setitem(
        sys.modules,
        "prometheus_fastapi_instrumentator",
        types.SimpleNamespace(Instrumentator=_DummyInstrumentator),
    )

    from app.api import main as api
    from fastapi.testclient import TestClient

    # Avoid heavy startup side effects in tests
    monkeypatch.setattr(api, "_load_model", lambda: None)
    monkeypatch.setattr(api, "_load_baseline", lambda: None)
    api.MODEL = object()

    with TestClient(api.app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


def test_api_predict_json_smoke(monkeypatch):
    class _DummyInstrumentator:
        def instrument(self, app):
            return self

        def expose(self, app):
            return self

    monkeypatch.setitem(
        sys.modules,
        "prometheus_fastapi_instrumentator",
        types.SimpleNamespace(Instrumentator=_DummyInstrumentator),
    )

    from app.api import main as api
    from fastapi.testclient import TestClient

    class _DummyModel:
        def infer_image(self, bgr, input_size=518):
            h, w = bgr.shape[:2]
            return np.full((h, w), 2.0, dtype=np.float32)

    monkeypatch.setattr(api, "_load_model", lambda: None)
    monkeypatch.setattr(api, "_load_baseline", lambda: None)
    api.MODEL = _DummyModel()

    # tiny valid RGB image
    img = Image.new("RGB", (32, 32), color=(120, 80, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    with TestClient(api.app) as client:
        r = client.post(
            "/predict/json",
            files={"file": ("sample.png", buf.getvalue(), "image/png")},
        )
        assert r.status_code == 200
        data = r.json()
        assert "depth_colormap_b64" in data
        assert "depth_gray_b64" in data
        assert "point_cloud" in data
        assert data["point_cloud"]["n_points"] >= 0
