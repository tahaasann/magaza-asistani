"""Niyet sınıflandırıcısının doğruluğunu ve maliyetini thinking_level'a göre ölçer (MA-005).

Çalıştırma:
  uv run python scripts/eval_intent.py --levels minimal       # tek seviye koş, sonucu kaydet
  uv run python scripts/eval_intent.py --report               # kayıtlı sonuçlardan tablo

Her seviyenin sonucu evals/results/intent_<seviye>.json dosyasına yazılır; böylece farklı
günlerde (ya da farklı anahtarlarla) koşulan seviyeler tek tabloda birleşir. Anahtar .env'den
okunur; script anahtarı asla yazdırmaz.

Free Tier dakikada 5 ve günde 20 istekle sınırlı (429 mesajlarından, 2026-11-11). Bu yüzden
çağrılar --rpm ile aralıklanır ve günlük kota dolunca koşu erken durdurulur.
"""

import argparse
import io
import json
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from magaza_asistani.intent import classify_intent
from magaza_asistani.llm import DEFAULT_MODEL, TIMEOUT_SECONDS, GeminiAPIError, ThinkingLevel

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "evals" / "intent_cases.json"
RESULTS_DIR = ROOT / "evals" / "results"
# "default" = thinking_level hiç gönderilmez, modelin kendi varsayılanı ölçülür.
LEVELS: dict[str, ThinkingLevel | None] = {"default": None, "minimal": "minimal", "low": "low"}


class Pacer:
    """İki çağrının başlangıcı arasında en az ``interval`` saniye bırakır.

    Tek bir Pacer tüm seviyeler için paylaşılır; böylece bir seviyenin son çağrısı ile
    sonrakinin ilk çağrısı arasında da beklenir (PR #4 review #2). ``clock`` ve ``sleep``
    testlerde sahte saatle değiştirilebilsin diye parametredir.
    """

    def __init__(
        self,
        interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval = interval
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        """Gerekirse bekler, sonra bu çağrının başlangıç zamanını not eder."""
        if self._last is not None:
            remaining = self._last + self.interval - self._clock()
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()


def load_records(name: str) -> dict[str, dict[str, Any]]:
    """Bir seviyenin kayıtlı sonuçlarını id'ye göre döndürür (dosya yoksa boş)."""
    path = RESULTS_DIR / f"intent_{name}.json"
    if not path.exists():
        return {}
    return {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))["records"]}


def run_level(
    cases: list[dict],
    name: str,
    client: httpx.Client,
    pacer: Pacer,
    previous: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Soruları tek seviyede sorar; her sorudan sonra dosyaya yazar.

    ``previous`` içinde başarılı sonucu olan sorular yeniden sorulmaz (--resume). Her soru
    sonrası kaydetmek, koşu yarıda çökerse elde edilen sonuçların kaybolmasını önler.
    """
    records: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        old = previous.get(case["id"])
        if old is not None and old["label"] is not None:
            records.append(old)
            continue
        record: dict[str, Any] = {"id": case["id"], "expected": case["expected"], "label": None}
        # Kota aşımını önlemek için çağrılar arasında bekleriz; bekleme süreye sayılmaz.
        pacer.wait()
        start = time.perf_counter()
        try:
            result = classify_intent(case["question"], thinking_level=LEVELS[name], client=client)
        except ValueError as exc:
            record["error"] = f"geçersiz cevap: {exc}"
        except httpx.HTTPError as exc:
            # Zaman aşımı, bağlantı kopması gibi ağ hataları koşuyu çökertmemeli.
            record["error"] = f"ağ hatası: {type(exc).__name__}"
        except GeminiAPIError as exc:
            record["error"] = f"API {exc.status_code}: {exc.message}"
            if exc.status_code == 429 and "per day" in exc.message:
                # Günlük kota dolduysa kalan çağrılar da reddedilir; boşuna denemeyiz.
                records.append(record)
                records.extend(
                    previous.get(rest["id"]) or _skipped(rest) for rest in cases[i + 1 :]
                )
                save(name, records)
                print(f"Günlük kota doldu: {exc.message}", flush=True)
                return records
        else:
            response = result.response
            record.update(
                label=result.intent.label,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                thought_tokens=response.thought_tokens,
                seconds=round(time.perf_counter() - start, 2),
            )
        records.append(record)
        # Henüz sorulmamış sorular da dosyada "atlandı" olarak durur; tablo hep 20 satır sayar.
        save(name, records + [previous.get(c["id"]) or _skipped(c) for c in cases[i + 1 :]])
        status = record["label"] or f"HATA ({record['error']})"
        print(f"  {case['id']}: {status} (beklenen {case['expected']})", flush=True)
    return records


def _skipped(case: dict) -> dict[str, Any]:
    """Sorulmadan kalan bir sorunun kaydı."""
    return {
        "id": case["id"],
        "expected": case["expected"],
        "label": None,
        "error": "atlandı: sorulmadı",
    }


def save(name: str, records: list[dict[str, Any]]) -> Path:
    """Bir seviyenin kayıtlarını model ve zaman bilgisiyle JSON dosyasına yazar."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"intent_{name}.json"
    payload = {
        "level": name,
        "model": DEFAULT_MODEL,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "records": records,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def report() -> None:
    """Kayıtlı tüm seviyeleri tek tabloda gösterir."""
    print(
        f"{'seviye':<10}{'doğruluk':>10}{'başarılı':>10}{'hata':>6}"
        f"{'ort. in':>9}{'ort. out':>9}{'ort. thought':>14}{'ort. sn':>9}  koşu zamanı"
    )
    misses: dict[str, list[str]] = {}
    for name in LEVELS:
        path = RESULTS_DIR / f"intent_{name}.json"
        if not path.exists():
            print(f"{name:<10}{'(henüz koşulmadı)':>30}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data["records"]
        ok = [r for r in records if r["label"] is not None]
        correct = sum(r["label"] == r["expected"] for r in records)
        n_ok = len(ok) or 1  # sıfıra bölmemek için

        # Doğruluk tüm sorulara bölünür (hata = yanlış); ortalamalar ise yalnız başarılı
        # çağrılara bölünür, yoksa başarısız çağrılar ortalamayı yapay düşürür.
        def avg(key: str, rows: list[dict[str, Any]] = ok, n: int = n_ok) -> float:
            return sum(r[key] for r in rows) / n

        print(
            f"{name:<10}{correct / len(records):>10.0%}{len(ok):>10}{len(records) - len(ok):>6}"
            f"{avg('input_tokens'):>9.0f}{avg('output_tokens'):>9.1f}"
            f"{avg('thought_tokens'):>14.0f}{avg('seconds'):>9.2f}  {data['run_at']}"
        )
        misses[name] = [
            f"{r['id']}: " + (r.get("error") or f"beklenen {r['expected']}, gelen {r['label']}")
            for r in records
            if r["label"] != r["expected"]
        ]
    print("\nYanlışlar:")
    for name, items in misses.items():
        print(f"- {name}: {'; '.join(items) if items else 'yok'}")


def main() -> None:
    # Windows konsolu varsayılan olarak cp1252 kullanır ve "ı", "ş" gibi harfleri basamaz.
    # En başta yapılır: --help de Türkçe metin yazdırır. sys.stdout'un tipi genel TextIO'dur;
    # reconfigure yalnız TextIOWrapper'da var. isinstance hem pyright'a tipi kanıtlar hem de
    # stdout başka bir nesneyle değiştirildiyse (ör. test yakalaması) çökmeyi önler.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--levels", nargs="+", choices=list(LEVELS), help="koşulacak seviyeler")
    parser.add_argument("--report", action="store_true", help="yalnız kayıtlı sonuçları göster")
    parser.add_argument(
        "--resume", action="store_true", help="kayıtta başarılı olan soruları yeniden sorma"
    )
    parser.add_argument(
        "--rpm", type=float, default=5, help="dakikadaki en fazla istek (0: sınırsız)"
    )
    args = parser.parse_args()
    if args.report:
        report()
        return
    if not args.levels:
        parser.error("--levels ya da --report ver")

    # 5 rpm için 12 sn; küçük bir pay ekleriz ki saat farkları sınırı aşmasın.
    delay = 60 / args.rpm + 0.5 if args.rpm > 0 else 0.0
    load_dotenv()
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    print(f"{len(cases)} soru x {len(args.levels)} seviye, çağrı arası {delay:.1f} sn\n")
    # Tüm seviyeler tek Pacer'ı paylaşır: seviye geçişinde de dakika sınırı korunur.
    pacer = Pacer(delay)
    # Tek bağlantı havuzu: her çağrıda yeni TCP/TLS bağlantısı açmayız.
    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        for name in args.levels:
            print(f"[{name}]", flush=True)
            previous = load_records(name) if args.resume else {}
            run_level(cases, name, client, pacer, previous)
            print(f"Kaydedildi: evals/results/intent_{name}.json\n", flush=True)
    report()


if __name__ == "__main__":
    main()
