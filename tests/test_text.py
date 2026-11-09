import pytest

from magaza_asistani.text import turkish_lower


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("İSTANBUL", "istanbul"),
        ("IĞDIR", "ığdır"),
        ("Şubat ÇİĞ Ürün IŞIK", "şubat çiğ ürün ışık"),
        ("Mağaza 42: İADE!", "mağaza 42: iade!"),
        ("", ""),
    ],
)
def test_turkish_lower(text: str, expected: str) -> None:
    assert turkish_lower(text) == expected


def test_decomposed_dotted_capital_i() -> None:
    assert turkish_lower("İSTANBUL") == "istanbul"


def test_result_has_no_combining_dot() -> None:
    assert "̇" not in turkish_lower("İSTANBUL")
