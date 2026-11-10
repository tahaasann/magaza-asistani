"""Gerçek Gemini çağrısı ile temperature farkını gözlemlemek için deneme scripti.

Çalıştırma: uv run python scripts/try_llm.py
Anahtar .env dosyasındaki GEMINI_API_KEY'den okunur; script anahtarı asla yazdırmaz.
"""

import sys

from dotenv import load_dotenv

from magaza_asistani.llm import ask

SYSTEM = "Sen bir market zincirinin mağaza asistanısın. Tek cümleyle cevap ver."
PROMPT = "Yeni açılan bir şubemiz için kısa ve akılda kalıcı bir slogan öner."
RUNS = 3


def main() -> None:
    # .env içeriğini ortam değişkenlerine yükler; değerleri ekrana basmaz.
    load_dotenv()
    # Windows konsolu varsayılan olarak cp1252 kullanır ve "ı", "ş" gibi harfleri basamaz.
    sys.stdout.reconfigure(encoding="utf-8")
    for temperature in (0.0, 1.0):
        print(f"\n=== temperature={temperature} ===")
        for run in range(1, RUNS + 1):
            result = ask(PROMPT, system=SYSTEM, temperature=temperature)
            print(f"{run}. {result.text.strip()}")
            print(f"   (girdi: {result.input_tokens} token, çıktı: {result.output_tokens} token)")


if __name__ == "__main__":
    main()
