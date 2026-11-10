"""llm.ask için testler. Ağa hiç çıkılmaz: HTTP katmanı httpx.MockTransport ile sahtelenir."""

import json
from collections.abc import Callable

import httpx
import pytest

from magaza_asistani.llm import DEFAULT_MODEL, GeminiAPIError, LLMResponse, ask

FAKE_KEY = "test-key-123"

OK_BODY = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "Merhaba!"}]}}],
    "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3, "totalTokenCount": 10},
}


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
    assert result == LLMResponse(text="Merhaba!", input_tokens=7, output_tokens=3)


def test_posts_to_model_endpoint_with_key_in_header_only() -> None:
    client, seen = make_client()
    ask("Selam", client=client)
    request = seen[0]
    assert request.method == "POST"
    assert request.url.path == f"/v1beta/models/{DEFAULT_MODEL}:generateContent"
    assert request.headers["x-goog-api-key"] == FAKE_KEY
    # Anahtar URL'ye sızmamalı.
    assert FAKE_KEY not in str(request.url)


def test_body_contains_prompt_and_temperature() -> None:
    client, seen = make_client()
    ask("İade nasıl yapılır?", temperature=0.7, client=client)
    body = sent_json(seen[0])
    assert body["contents"][0]["parts"][0]["text"] == "İade nasıl yapılır?"
    assert body["generationConfig"]["temperature"] == 0.7


@pytest.mark.parametrize(
    ("system", "check"),
    [
        (None, lambda body: "systemInstruction" not in body),
        (
            "Kısa cevap ver.",
            lambda body: body["systemInstruction"]["parts"][0]["text"] == "Kısa cevap ver.",
        ),
    ],
)
def test_system_instruction_only_when_given(
    system: str | None, check: Callable[[dict], bool]
) -> None:
    client, seen = make_client()
    ask("Selam", system=system, client=client)
    assert check(sent_json(seen[0]))


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
    # 429'un asıl nedeni (ör. free tier kotası 0) hata mesajında görünmeli.
    body = {"error": {"status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded, limit: 0"}}
    client, _ = make_client(status=429, body=body)
    with pytest.raises(GeminiAPIError, match="limit: 0") as exc_info:
        ask("Selam", client=client)
    assert exc_info.value.message == "Quota exceeded, limit: 0"


def test_http_error_with_non_json_body() -> None:
    # Proxy'nin HTML hata sayfası gibi JSON olmayan gövdede de anlaşılır hata vermeli.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="<html>Bad Gateway</html>")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(GeminiAPIError, match="502") as exc_info:
        ask("Selam", client=client)
    assert exc_info.value.message == ""


def test_no_candidates_raises() -> None:
    client, _ = make_client(body={"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(ValueError, match="SAFETY"):
        ask("Selam", client=client)


def test_empty_text_raises_with_finish_reason() -> None:
    # Yanıt token sınırında kesilip metin boş geldiyse sessizce "" dönmemeli.
    body = {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]}
    client, _ = make_client(body=body)
    with pytest.raises(ValueError, match="MAX_TOKENS"):
        ask("Selam", client=client)
