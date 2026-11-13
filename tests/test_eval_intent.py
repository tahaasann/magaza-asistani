"""scripts/eval_intent.py için testler: ölçüm aracının kendisi de koddur (PR #4 review #3).

Ağa çıkılmaz; HTTP katmanı MockTransport ile sahtelenir, sonuç dosyaları geçici klasöre yazılır.
"""

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import httpx
import pytest

ROOT = Path(__file__).parents[1]
INTENT_OK = json.loads(
    (ROOT / "tests" / "fixtures" / "interaction_intent_ok.json").read_text(encoding="utf-8")
)
DAILY_429 = {"error": {"message": "Rate limit exceeded (limit: 20 requests per day on Free Tier)."}}
CASES = [{"id": f"q{i}", "question": f"Soru {i}", "expected": "procedure"} for i in range(4)]


def load_script() -> ModuleType:
    """scripts/ bir paket değil; modülü dosya yolundan yükleriz."""
    spec = importlib.util.spec_from_file_location(
        "eval_intent", ROOT / "scripts" / "eval_intent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    module = load_script()
    # Gerçek evals/results/ dosyalarının üzerine yazmayalım.
    monkeypatch.setattr(module, "RESULTS_DIR", tmp_path)
    return module


def make_client(
    responder: Callable[[int], httpx.Response],
) -> tuple[httpx.Client, list[httpx.Request]]:
    """Kaçıncı istek olduğuna göre yanıt veren sahte istemci."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responder(len(seen))

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def no_wait(script: ModuleType) -> object:
    return script.Pacer(0.0, sleep=lambda _: None)


def test_pacer_waits_between_calls_even_across_levels(script: ModuleType) -> None:
    # Sahte saat: zaman yalnız sleep çağrılınca ilerler.
    now = [100.0]
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    pacer = script.Pacer(12.5, clock=lambda: now[0], sleep=sleep)
    pacer.wait()  # 1. seviyenin son çağrısı: beklemez
    now[0] += 2.0  # çağrı 2 sn sürdü
    pacer.wait()  # 2. seviyenin ilk çağrısı: kalan 10.5 sn beklenmeli
    assert slept == [pytest.approx(10.5)]


def test_daily_quota_stops_run_and_marks_rest_skipped(script: ModuleType, tmp_path: Path) -> None:
    client, seen = make_client(
        lambda n: (
            httpx.Response(200, json=INTENT_OK) if n == 1 else httpx.Response(429, json=DAILY_429)
        )
    )
    records = script.run_level(CASES, "minimal", client, no_wait(script), {})
    # 1 başarılı + 1 kota hatası; kalan 2 soru hiç gönderilmemeli.
    assert len(seen) == 2
    assert [r["label"] for r in records] == ["procedure", None, None, None]
    assert records[1]["error"].startswith("API 429")
    assert all(r["error"] == "atlandı: sorulmadı" for r in records[2:])
    # Erken durunca da sonuç dosyaya yazılmış olmalı (kaldığı yer kaybolmasın).
    saved = json.loads((tmp_path / "intent_minimal.json").read_text(encoding="utf-8"))
    assert len(saved["records"]) == 4


def test_resume_does_not_ask_answered_questions_again(script: ModuleType) -> None:
    client, seen = make_client(lambda n: httpx.Response(200, json=INTENT_OK))
    previous = {
        "q0": {"id": "q0", "expected": "procedure", "label": "procedure"},
        "q1": {"id": "q1", "expected": "procedure", "label": None, "error": "API 429: ..."},
    }
    records = script.run_level(CASES, "minimal", client, no_wait(script), previous)
    # q0 zaten cevaplıydı: yalnız q1, q2, q3 sorulur.
    assert len(seen) == 3
    assert records[0] is previous["q0"]
    assert all(r["label"] == "procedure" for r in records)


def test_network_error_is_recorded_not_raised(script: ModuleType) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("zaman aşımı", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = script.run_level(CASES[:1], "minimal", client, no_wait(script), {})
    assert records[0]["error"] == "ağ hatası: ReadTimeout"
