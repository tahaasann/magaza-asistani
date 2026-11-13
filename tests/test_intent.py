"""intent.classify_intent için testler. Ağa çıkılmaz; HTTP katmanı MockTransport ile sahtelenir.

``interaction_intent_ok.json``, response_format ile yapılan gerçek bir probe çağrısının yanıtıdır
(2026-11-11): JSON cevap, model_output adımının text öğesinde string olarak gelir.
"""

import json
from collections import Counter
from pathlib import Path
from typing import get_args

import httpx
import pytest

from magaza_asistani.intent import Intent, IntentLabel, classify_intent
from magaza_asistani.llm import ThinkingLevel

FIXTURES = Path(__file__).parent / "fixtures"
EVAL_CASES = Path(__file__).parents[1] / "evals" / "intent_cases.json"
INTENT_OK = json.loads((FIXTURES / "interaction_intent_ok.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")


def make_client(body: dict) -> tuple[httpx.Client, list[httpx.Request]]:
    """Gelen istekleri kaydeden ve sabit bir yanıt dönen sahte istemci kurar."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def body_with_text(text: str) -> dict:
    """Model cevabı olarak verilen metni taşıyan, tamamlanmış en küçük yanıt gövdesi."""
    return {
        "status": "completed",
        "steps": [{"type": "model_output", "content": [{"type": "text", "text": text}]}],
        "usage": {"total_input_tokens": 1, "total_output_tokens": 1},
    }


def test_parses_label_and_keeps_token_usage() -> None:
    client, _ = make_client(INTENT_OK)
    result = classify_intent("Soğutucu arızalandığında ne yapmam gerekiyor?", client=client)
    assert result.intent == Intent(label="procedure")
    # Eval maliyeti ölçebilsin diye token bilgisi kaybolmamalı (gerçek probe değerleri).
    assert result.response.output_tokens == 5
    assert result.response.thought_tokens == 218


def test_request_carries_schema_and_keeps_store_false() -> None:
    client, seen = make_client(INTENT_OK)
    classify_intent("İade nasıl yapılır?", client=client)
    body = json.loads(seen[0].content)
    fmt = body["response_format"]
    assert fmt["mime_type"] == "application/json"
    assert fmt["schema"] == Intent.model_json_schema()
    assert fmt["schema"]["properties"]["label"]["enum"] == [
        "procedure",
        "data",
        "action",
        "unknown",
    ]
    assert body["store"] is False
    assert body["generation_config"]["temperature"] == 0.0


@pytest.mark.parametrize("level", [None, "minimal", "low"])
def test_thinking_level_is_forwarded(level: ThinkingLevel | None) -> None:
    client, seen = make_client(INTENT_OK)
    classify_intent("Dün kaç ekmek sattık?", thinking_level=level, client=client)
    config = json.loads(seen[0].content)["generation_config"]
    if level is None:
        assert "thinking_level" not in config
    else:
        assert config["thinking_level"] == level


@pytest.mark.parametrize(
    "text",
    [
        "procedure",  # JSON değil, düz metin
        '{"label": "refund"}',  # şemada olmayan etiket
        '{"intent": "data"}',  # zorunlu alan yok
    ],
)
def test_invalid_model_output_raises(text: str) -> None:
    # Bozuk cevap sessizce "unknown" yapılmamalı; eval'de ayrı sayılabilmesi için hata vermeli.
    client, _ = make_client(body_with_text(text))
    with pytest.raises(ValueError, match="şemaya uymayan"):
        classify_intent("Selam", client=client)


def test_eval_cases_are_well_formed() -> None:
    # Eval setindeki bir yazım hatası (ör. "procedur") ölçümü sessizce bozar; önce seti sınarız.
    cases = json.loads(EVAL_CASES.read_text(encoding="utf-8"))
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)), "id'ler tekil olmalı"
    labels = Counter(case["expected"] for case in cases)
    assert set(labels) == set(get_args(IntentLabel))
    assert min(labels.values()) >= 5, "her etiketten en az 5 soru olmalı"


def test_default_thinking_level_is_minimal() -> None:
    # Eval kararı (minimal: aynı doğruluk, 0 düşünme token'ı) varsayılan olarak gönderilmeli.
    client, seen = make_client(INTENT_OK)
    classify_intent("Fire kaydı nasıl girilir?", client=client)
    assert json.loads(seen[0].content)["generation_config"]["thinking_level"] == "minimal"


def test_invalid_output_error_is_truncated() -> None:
    # Ham cevap kişisel veri taşıyabilir; hata mesajı yalnız kısa bir önizleme içermeli.
    client, _ = make_client(body_with_text("x" * 500))
    with pytest.raises(ValueError) as exc_info:
        classify_intent("Selam", client=client)
    assert "x" * 101 not in str(exc_info.value)
    assert "x" * 100 in str(exc_info.value)
