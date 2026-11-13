"""Mağaza çalışanının sorusunu hangi uzmana gideceğine göre etiketleyen sınıflandırıcı (MA-005).

Orkestratörün (Aşama 2) atasıdır: soru ``procedure`` (prosedür ajanı), ``data`` (veri ajanı),
``action`` (aksiyon ajanı) ya da ``unknown`` olarak etiketlenir. Model serbest metin değil,
JSON şemasına uyan bir cevap döndürür (structured output); biz de cevabı Pydantic ile yeniden
doğrularız, çünkü modelin şemaya uyduğuna körü körüne güvenmeyiz.
"""

from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ValidationError

from magaza_asistani.llm import DEFAULT_MODEL, LLMResponse, ThinkingLevel, ask

IntentLabel = Literal["procedure", "data", "action", "unknown"]

# Eval (2026-11-11): minimal 20/20 doğru ve 0 düşünme token'ı; default ve low aynı doğrulukta
# ~250 düşünme token'ı harcadı. Kalite çıtasını tutan en ucuz ayar varsayılan olur.
DEFAULT_THINKING_LEVEL: ThinkingLevel = "minimal"
# Şemaya uymayan cevabın hata mesajında gösterilecek en fazla karakter.
ERROR_PREVIEW_CHARS = 100

SYSTEM_PROMPT = """\
Sen bir market zincirinin mağaza asistanında soru yönlendiricisisin.
Mağaza çalışanının sorusunu tek bir etikete ata:
- procedure: Bir işin nasıl yapılacağı, kural ya da prosedür sorusu \
(iade, fire kaydı, son kullanma tarihi, kasa açılış/kapanış, arıza olunca yapılacaklar).
- data: Satış, stok, fire gibi rakamlar ya da rapor isteyen soru.
- action: Asistandan sistemde bir işlem yapmasını isteme \
(arıza kaydı aç, sipariş oluştur, görev ata).
- unknown: Mağaza işiyle ilgisiz, belirsiz ya da bu üçüne uymayan soru.
Emin değilsen unknown seç."""


class Intent(BaseModel):
    """Modelin döndürmesi gereken JSON'un şekli; şema da bu sınıftan üretilir."""

    label: IntentLabel


@dataclass(frozen=True)
class Classification:
    """Etiket ve çağrının ham yanıtı (eval token maliyetini buradan okur)."""

    intent: Intent
    response: LLMResponse


def _response_format() -> dict[str, Any]:
    """Interactions API'nin beklediği response_format nesnesini Intent şemasından kurar."""
    return {
        "type": "text",
        "mime_type": "application/json",
        "schema": Intent.model_json_schema(),
    }


def classify_intent(
    question: str,
    *,
    thinking_level: ThinkingLevel | None = DEFAULT_THINKING_LEVEL,
    model: str = DEFAULT_MODEL,
    client: httpx.Client | None = None,
) -> Classification:
    """Soruyu dört etiketten birine atar.

    ``thinking_level=None`` verilirse alan gönderilmez ve modelin kendi varsayılanı kullanılır
    (eval bunu "default" seviyesi olarak ölçer).

    Şemaya uymayan cevap sessizce ``unknown`` yapılmaz, ``ValueError`` fırlatılır: böylece
    eval'de "model yanlış etiketledi" ile "model bozuk cevap verdi" birbirinden ayrılır.
    """
    response = ask(
        question,
        system=SYSTEM_PROMPT,
        # Sınıflandırmada yaratıcılık istemeyiz; aynı soru aynı etiketi almalı.
        temperature=0.0,
        model=model,
        response_format=_response_format(),
        thinking_level=thinking_level,
        client=client,
    )
    try:
        intent = Intent.model_validate_json(response.text)
    except ValidationError as exc:
        # Ham cevap kullanıcının sorusunu tekrar ediyor olabilir (kişisel veri, KVKK);
        # log'lara düşmesin diye yalnız başını taşırız (PR #4 review #5).
        preview = response.text[:ERROR_PREVIEW_CHARS]
        raise ValueError(f"Model şemaya uymayan cevap döndü: {preview!r}") from exc
    return Classification(intent=intent, response=response)
