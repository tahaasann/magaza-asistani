"""Türkçe metinler için yardımcı fonksiyonlar."""

import unicodedata


def turkish_lower(text: str) -> str:
    """Metni Türkçe kurallara göre küçük harfe çevirir (İ -> i, I -> ı).

    ``str.lower()`` dile göre davranmaz: ``I`` harfini ``i`` yapar, ``İ`` harfini ise
    ``i`` ve görünmez bir nokta işaretine böler. Bu da Türkçe metinde eşleşmeyi bozar.
    """
    composed = unicodedata.normalize("NFC", text)
    return composed.replace("İ", "i").replace("I", "ı").lower()
