"""Fast checks of the pieces every metric depends on (no GPU, no audio). Run: pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from numbers_es import find_numbers  # noqa: E402
from text_norm import normalize  # noqa: E402
from validate import validate  # noqa: E402
from verify import SIN_VERIFICAR, verify_call, verify_interview  # noqa: E402


def nums(s):
    return [v for v, _, _ in find_numbers(normalize(s))]


def test_normalize_digits_and_words_match():
    assert normalize("6.500 soles") == normalize("seis mil quinientos soles")
    assert normalize("Ochenta y un soles") == normalize("81 soles")
    assert normalize("certificación C1") == "certificacion c1"


def test_spoken_numbers():
    assert nums("casi tres años, no, son dos años y medio") == [3, 2.5]
    assert nums("entre nueve y diez mil, diría 9.800") == [9, 10000, 9800]
    assert nums("el total es de 85, 88 soles") == [85, 88]
    assert nums("departamento mil ciento cuatro") == [1104]
    assert nums("una pizza y un pan") == []


def test_verification_rejects_unsaid_values():
    t = "Mi último puesto es analista. Pido seis mil soles. Mi inglés es intermedio."
    v = verify_interview({"anios_experiencia": 4, "ultimo_puesto": "analista", "tecnologias": ["Python"],
                          "pretension_salarial": 6000, "disponibilidad": None, "nivel_ingles": "intermedio"}, t)
    assert v["pretension_salarial"] == 6000 and v["nivel_ingles"] == "intermedio"
    assert v["anios_experiencia"] == SIN_VERIFICAR and v["tecnologias"] == SIN_VERIFICAR


def test_arithmetic_flags_wrong_total():
    t = "una americana familiar y una gaseosa. el total es de sesenta soles. pago con yape"
    fields = {"items": [{"producto": "americana familiar", "cantidad": 1, "precio_unitario": 45},
                        {"producto": "gaseosa 1.5 L", "cantidad": 1, "precio_unitario": 9}],
              "promocion": None, "direccion_entrega": None, "metodo_pago": "yape", "total_dicho": 60}
    val = validate(verify_call(fields, t), t)
    assert val["total_calculado"] == 54 and val["estado"] == "no_cuadra" and val["diagnostico"] == "error_hablante"
