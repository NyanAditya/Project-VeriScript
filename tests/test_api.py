"""REST API tests with the stub cascade injected (no model downloads)."""

import io
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

import app.api as api

AI_TEXT = " ".join(["the cat sat on the mat"] * 12)
HUMAN_TEXT = " ".join(f"token{j}" for j in range(80))


@pytest.fixture()
def client(stub_cascade, monkeypatch):
    monkeypatch.setitem(api._state, "detector", stub_cascade)
    monkeypatch.setenv("VERISCRIPT_ADMIN_TOKEN", "secret")
    return TestClient(api.app)


def test_detect(client):
    r = client.post("/api/v1/detect", json={"text": AI_TEXT})
    assert r.status_code == 200
    js = r.json()
    assert js["label"] == "ai_generated" and js["route"] == "fast_path"
    assert set(js["latency_ms"]) >= {"heuristic", "transformer"}
    assert js["explanation"]["top_features"]


def test_detect_mode_does_not_leak(client, stub_cascade):
    before = stub_cascade.operating_point
    r = client.post("/api/v1/detect", json={"text": HUMAN_TEXT, "mode": "thorough"})
    assert r.json()["route"] == "escalated"
    assert stub_cascade.operating_point == before


def test_detect_file(client):
    r = client.post("/api/v1/detect/file", files={"file": ("a.txt", AI_TEXT.encode())})
    assert r.status_code == 200 and r.json()["id"] == "a.txt"


def test_batch_csv_and_zip(client):
    csv_bytes = f"id,text\nx,{AI_TEXT}\ny,short\n".encode()
    job = client.post("/api/v1/batch", files={"file": ("b.csv", csv_bytes)}).json()
    for _ in range(50):
        st = client.get(f"/api/v1/batch/{job['job_id']}").json()
        if st["status"] == "completed":
            break
        time.sleep(0.05)
    assert st["status"] == "completed"
    assert st["summary"]["n_documents"] == 2 and st["summary"]["n_undetermined"] == 1

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("one.txt", HUMAN_TEXT)
        z.writestr("two.txt", AI_TEXT)
    job = client.post("/api/v1/batch", files={"file": ("b.zip", buf.getvalue())}).json()
    assert job["n_documents"] == 2


def test_config_admin_only(client):
    assert client.get("/api/v1/config").json()["operating_point"] in ("balanced", "custom")
    assert client.put("/api/v1/config", json={"operating_point": "thorough"}).status_code == 403
    r = client.put("/api/v1/config", json={"operating_point": "thorough"}, headers={"X-Admin-Token": "secret"})
    assert r.status_code == 200 and r.json()["operating_point"] == "thorough"
    r = client.put("/api/v1/config", json={"t_low": 0.2, "t_high": 0.8}, headers={"X-Admin-Token": "secret"})
    assert r.json()["operating_point"] == "custom"
    client.put("/api/v1/config", json={"operating_point": "balanced"}, headers={"X-Admin-Token": "secret"})


def test_health(client):
    assert client.get("/api/v1/health").json()["models_loaded"] is True


def test_unknown_job(client):
    assert client.get("/api/v1/batch/nope").status_code == 404
