"""Gemini Interactions API'ye SDK kullanmadan, ham HTTP ile istek atan en küçük istemci.

Amaç, framework'lerin arka planda ne yaptığını görmek: bir LLM çağrısı aslında
JSON gövdeli tek bir HTTP POST isteğidir. Haziran 2026'dan beri varsayılan arayüz
Interactions API'dir; eski ``generateContent`` legacy sayılır (MA-007).
"""

import os
from dataclasses import dataclass
from typing import Any, Literal

import httpx

# 3.5 Flash-Lite ve 3.1 Flash-Lite temperature'ı pratikte yok sayıyor (2026-11-10, generateContent);
# 3 Flash Preview tepki veriyor. Interactions API'de de doğrulandı (2026-11-10): temperature 0'da
# 3 cevabın 3'ü aynı, 2.0'da 3'ü farklı. Preview modeller değişebilir: kaldırılırsa güncellenir.
DEFAULT_MODEL = "gemini-3-flash-preview"
API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
API_KEY_ENV = "GEMINI_API_KEY"
TIMEOUT_SECONDS = 30.0

# Modelin cevap vermeden önce ne kadar "düşüneceği".
# Gönderilmezse modelin kendi varsayılanı geçerli olur (dokümanda yazmıyor, MA-005'te ölçülür).
ThinkingLevel = Literal["minimal", "low", "medium", "high"]


@dataclass(frozen=True)
class LLMResponse:
    """Modelin yanıt metni ve bu çağrının token kullanımı."""

    text: str
    input_tokens: int
    output_tokens: int
    # Düşünme token'ları cevapta görünmez ama faturaya girer; çoğu zaman çıktı token'ından fazladır.
    thought_tokens: int = 0


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


def _build_body(
    model: str,
    prompt: str,
    system: str | None,
    temperature: float,
    response_format: dict[str, Any] | None = None,
    thinking_level: ThinkingLevel | None = None,
) -> dict[str, Any]:
    """İstek gövdesini (JSON) Interactions API'nin beklediği biçimde kurar."""
    generation_config: dict[str, Any] = {"temperature": temperature}
    if thinking_level is not None:
        generation_config["thinking_level"] = thinking_level
    body: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "generation_config": generation_config,
        # Varsayılan store=true her isteği Google tarafında saklar. Biz durumsuz çalışıyoruz
        # (previous_interaction_id kullanmıyoruz), bu yüzden veri bırakmayız (KVKK).
        "store": False,
    }
    # System prompt yalnızca verildiyse eklenir; boş bir talimat göndermeyiz.
    if system is not None:
        body["system_instruction"] = system
    # Yapılandırılmış çıktı istenirse şema gövdenin üst seviyesine konur
    # (generation_config içine değil; dokümanda böyle).
    if response_format is not None:
        body["response_format"] = response_format
    return body


def _parse_response(data: dict[str, Any]) -> LLMResponse:
    """Yanıt gövdesinden metni ve token sayılarını çıkarır."""
    status = data.get("status")
    if status != "completed":
        # Ör. token sınırında kesilirse "incomplete", sunucu hatasında "failed" gelir.
        raise ValueError(f"Model yanıtı tamamlanmadı (status: {status})")
    # Yanıt adımlardan oluşur: "thought" (düşünme) adımlarında kullanıcıya metin yoktur;
    # cevap yalnızca "model_output" adımlarının text öğelerindedir.
    text = "".join(
        item.get("text", "")
        # `or []`: alan hiç yoksa da null gelirse de boş liste gibi davranırız.
        for step in data.get("steps") or []
        if step.get("type") == "model_output"
        for item in step.get("content") or []
        if item.get("type") == "text"
    )
    if not text:
        # Boş cevabı sessizce "" olarak döndürmeyiz.
        raise ValueError(f"Model boş yanıt döndü (status: {status})")
    usage = data.get("usage", {})
    return LLMResponse(
        text=text,
        input_tokens=usage.get("total_input_tokens", 0),
        output_tokens=usage.get("total_output_tokens", 0),
        thought_tokens=usage.get("total_thought_tokens", 0),
    )


def ask(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.0,
    *,
    model: str = DEFAULT_MODEL,
    response_format: dict[str, Any] | None = None,
    thinking_level: ThinkingLevel | None = None,
    client: httpx.Client | None = None,
) -> LLMResponse:
    """Modele tek bir soru sorar ve yanıtı döndürür.

    ``response_format`` ve ``thinking_level`` verilmezse gövdeye hiç girmez; böylece
    eski çağrılar aynen çalışır. ``client`` testlerde sahte (mock) bir HTTP katmanı
    vermek içindir; normal kullanımda boş bırakılır ve fonksiyon kendi istemcisini açıp kapatır.
    """
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV} tanımlı değil. Anahtarı .env dosyasına ekle.")

    # Anahtar URL'ye değil header'a konur: URL'ler log'lara düşer.
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    body = _build_body(model, prompt, system, temperature, response_format, thinking_level)

    if client is None:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as own_client:
            response = own_client.post(API_URL, headers=headers, json=body)
    else:
        response = client.post(API_URL, headers=headers, json=body)

    _raise_for_error(response)
    return _parse_response(response.json())
