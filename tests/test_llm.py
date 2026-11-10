"""llm.ask için testler. Ağa hiç çıkılmaz: HTTP katmanı httpx.MockTransport ile sahtelenir.

Örnek yanıtlar (fixtures/) Interactions API'ye atılan gerçek keşif çağrılarından alındı;
böylece testler tahmin ettiğimiz yapıyı değil, API'nin gerçekten döndürdüğü yapıyı sınar.
"""

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from magaza_asistani.llm import DEFAULT_MODEL, GeminiAPIError, LLMResponse, ask

FAKE_KEY = "test-key-123"
FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


OK_BODY = load_fixture("interaction_ok.json")


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Gerçek anahtar yerine sahte bir değer; .env hiç okunmaz.
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)


def make_client(
    status: int = 200, body: dict | None = None
) -> tuple[httpx.Client, list[httpx.Request]]:
    """Gelen istekleri kaydeden ve sabit bir yanıt dönen sahte istemci kurar."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=OK_BODY if body is None else body)

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def sent_json(request: httpx.Request) -> dict:
    return json.loads(request.content)


def test_returns_text_and_token_counts() -> None:
    client, _ = make_client()
    result = ask("Selam", client=client)
    assert result == LLMResponse(
        text="Tazelik ve uygun fiyatın yeni adresi mahallenizde sizi bekliyor.",
        input_tokens=31,
        output_tokens=18,
    )


def test_posts_to_interactions_endpoint_with_key_in_header_only() -> None:
    client, seen = make_client()
    ask("Selam", client=client)
    request = seen[0]
    assert request.method == "POST"
    assert request.url.path == "/v1beta/interactions"
    assert request.headers["x-goog-api-key"] == FAKE_KEY
    # Anahtar URL'ye sızmamalı.
    assert FAKE_KEY not in str(request.url)


def test_body_contains_model_prompt_and_temperature() -> None:
    client, seen = make_client()
    ask("İade nasıl yapılır?", temperature=0.7, client=client)
    body = sent_json(seen[0])
    assert body["model"] == DEFAULT_MODEL
    assert body["input"] == "İade nasıl yapılır?"
    assert body["generation_config"]["temperature"] == 0.7


def test_body_disables_server_side_storage() -> None:
    # store varsayılan olarak true; durumsuz çalıştığımız için istek saklanmamalı (KVKK).
    client, seen = make_client()
    ask("Selam", client=client)
    assert sent_json(seen[0])["store"] is False


@pytest.mark.parametrize(
    ("system", "check"),
    [
        (None, lambda body: "system_instruction" not in body),
        ("Kısa cevap ver.", lambda body: body["system_instruction"] == "Kısa cevap ver."),
    ],
)
def test_system_instruction_only_when_given(
    system: str | None, check: Callable[[dict], bool]
) -> None:
    client, seen = make_client()
    ask("Selam", system=system, client=client)
    assert check(sent_json(seen[0]))


def test_only_model_output_text_is_returned() -> None:
    # Düşünme adımı metin taşısa bile kullanıcıya gösterilmemeli; birden çok text öğesi birleşir.
    body = {
        "status": "completed",
        "steps": [
            {"type": "thought", "content": [{"type": "text", "text": "gizli düşünce"}]},
            {
                "type": "model_output",
                "content": [
                    {"type": "text", "text": "Merhaba "},
                    {"type": "text", "text": "dünya"},
                ],
            },
        ],
        "usage": {"total_input_tokens": 1, "total_output_tokens": 2},
    }
    client, _ = make_client(body=body)
    assert ask("Selam", client=client).text == "Merhaba dünya"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY")
    client, seen = make_client()
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        ask("Selam", client=client)
    # Anahtar yoksa istek hiç gönderilmemeli.
    assert seen == []


@pytest.mark.parametrize("status", [400, 429, 500])
def test_http_error_raises_without_leaking_key(status: int) -> None:
    client, _ = make_client(status=status, body={"error": {"message": "hata"}})
    with pytest.raises(GeminiAPIError) as exc_info:
        ask("Selam", client=client)
    assert exc_info.value.status_code == status
    assert FAKE_KEY not in str(exc_info.value)


def test_http_error_carries_google_message() -> None:
    # Gerçek 404 yanıtı: Google'ın açıklaması (ör. yanlış model adı) hatada görünmeli.
    client, _ = make_client(status=404, body=load_fixture("interaction_error_404.json"))
    with pytest.raises(GeminiAPIError, match="not found") as exc_info:
        ask("Selam", client=client)
    assert exc_info.value.status_code == 404
    assert "gemini-bu-model-yok" in exc_info.value.message


def test_http_error_with_non_json_body() -> None:
    # Proxy'nin HTML hata sayfası gibi JSON olmayan gövdede de anlaşılır hata vermeli.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="<html>Bad Gateway</html>")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(GeminiAPIError, match="502") as exc_info:
        ask("Selam", client=client)
    assert exc_info.value.message == ""


@pytest.mark.parametrize("status", ["incomplete", "failed", None])
def test_not_completed_status_raises(status: str | None) -> None:
    # Ör. token sınırında kesilen yanıt "incomplete" gelir; yarım cevabı döndürmemeliyiz.
    client, _ = make_client(body={"status": status, "steps": []})
    with pytest.raises(ValueError, match=f"status: {status}"):
        ask("Selam", client=client)


@pytest.mark.parametrize(
    "steps",
    [
        [{"type": "thought", "signature": "x"}],  # yalnızca düşünme adımı
        None,  # steps alanı null
        [{"type": "model_output", "content": None}],  # content alanı null
    ],
)
def test_empty_text_raises(steps: list | None) -> None:
    # Metinsiz tamamlanmış yanıt: çökmemeli, sessizce "" de dönmemeli; açık hata vermeli.
    body = {"status": "completed", "steps": steps}
    client, _ = make_client(body=body)
    with pytest.raises(ValueError, match="boş yanıt"):
        ask("Selam", client=client)
