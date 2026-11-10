"""Gemini API'ye SDK kullanmadan, ham HTTP ile istek atan en küçük istemci.

Amaç, framework'lerin arka planda ne yaptığını görmek: bir LLM çağrısı aslında
JSON gövdeli tek bir HTTP POST isteğidir.
"""

import os
from dataclasses import dataclass
from typing import Any

import httpx

# 3.5 Flash-Lite ve 3.1 Flash-Lite temperature'ı pratikte yok sayıyor (2026-11-10 deneyi);
# 3 Flash Preview ise tepki veriyor. Preview modeller değişebilir: kaldırılırsa burası güncellenir.
DEFAULT_MODEL = "gemini-3-flash-preview"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
API_KEY_ENV = "GEMINI_API_KEY"
TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class LLMResponse:
    """Modelin yanıt metni ve bu çağrının token kullanımı."""

    text: str
    input_tokens: int
    output_tokens: int


class GeminiAPIError(RuntimeError):
    """API 4xx/5xx döndüğünde fırlatılır; Google'ın açıklamasını taşır, anahtarı asla taşımaz."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(f"Gemini API hatası {status_code}: {message or '(açıklama yok)'}")


def _raise_for_error(response: httpx.Response) -> None:
    """Yanıt başarısızsa, gövdedeki ``error.message`` ile birlikte GeminiAPIError fırlatır."""
    if not response.is_error:
        return
    try:
        data = response.json()
    except ValueError:
        # Gövde JSON değilse (ör. bir proxy'nin HTML hata sayfası) açıklamasız devam ederiz.
        data = None
    error = data.get("error", {}) if isinstance(data, dict) else {}
    message = error.get("message", "") if isinstance(error, dict) else ""
    raise GeminiAPIError(response.status_code, message)


def _build_body(prompt: str, system: str | None, temperature: float) -> dict[str, Any]:
    """İstek gövdesini (JSON) Gemini'nin beklediği biçimde kurar."""
    body: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    # System prompt yalnızca verildiyse eklenir; boş bir talimat göndermeyiz.
    if system is not None:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    return body


def _parse_response(data: dict[str, Any]) -> LLMResponse:
    """Yanıt gövdesinden metni ve token sayılarını çıkarır."""
    candidates = data.get("candidates") or []
    if not candidates:
        # Örneğin güvenlik filtresi istemi engellediyse aday yanıt gelmez.
        raise ValueError(f"Model yanıt üretmedi: {data.get('promptFeedback')}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts)
    if not text:
        # Token sınırı ya da güvenlik nedeniyle kesilen yanıtı sessizce "" olarak döndürmeyiz.
        reason = candidates[0].get("finishReason")
        raise ValueError(f"Model boş yanıt döndü (finishReason: {reason})")
    usage = data.get("usageMetadata", {})
    return LLMResponse(
        text=text,
        input_tokens=usage.get("promptTokenCount", 0),
        output_tokens=usage.get("candidatesTokenCount", 0),
    )


def ask(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.0,
    *,
    model: str = DEFAULT_MODEL,
    client: httpx.Client | None = None,
) -> LLMResponse:
    """Modele tek bir soru sorar ve yanıtı döndürür.

    ``client`` testlerde sahte (mock) bir HTTP katmanı vermek içindir; normal
    kullanımda boş bırakılır ve fonksiyon kendi istemcisini açıp kapatır.
    """
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV} tanımlı değil. Anahtarı .env dosyasına ekle.")

    # Anahtar URL'ye değil header'a konur: URL'ler log'lara düşer.
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    url = API_URL.format(model=model)
    body = _build_body(prompt, system, temperature)

    if client is None:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as own_client:
            response = own_client.post(url, headers=headers, json=body)
    else:
        response = client.post(url, headers=headers, json=body)

    _raise_for_error(response)
    return _parse_response(response.json())
