"""
Модуль правил ГОСТ Р 50577-2018 для государственных регистрационных знаков РФ.
"""

import re
from typing import Tuple

ALLOWED_LATIN_LETTERS = frozenset("ABEKMHOPCTYX")
DIGITS = frozenset("0123456789")
UNREADABLE_CHAR = "#"

CYRILLIC_TO_LATIN_MAP = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
    "а": "A", "в": "B", "е": "E", "к": "K", "м": "M", "н": "H",
    "о": "O", "р": "P", "с": "C", "т": "T", "у": "Y", "х": "X",
})

VALID_PLATE_TYPES = ("type1", "type1a", "type1b", "other")

GOST_TARGET_REGEX = re.compile(
    r"^[ABEKMHOPCTYX#][0-9#]{3}[ABEKMHOPCTYX#]{2}(?:[127#][0-9#]{2}|[0-9#]{2})$"
)


def normalize_plate_string(raw_text: str) -> str:
    if not raw_text:
        return ""
    cleaned = raw_text.strip().replace(" ", "").replace("-", "").upper()
    return cleaned.translate(CYRILLIC_TO_LATIN_MAP)


def is_valid_gost_target_mask(plate_num: str, max_hashes: int = 4) -> Tuple[bool, str]:
    norm = normalize_plate_string(plate_num)
    if len(norm) not in (8, 9):
        return False, f"Invalid length {len(norm)} (expected 8 or 9)"

    hash_count = norm.count(UNREADABLE_CHAR)
    if hash_count > max_hashes:
        return False, f"Too many unreadable characters '#': {hash_count} > {max_hashes}"

    if hash_count == len(norm):
        return False, "Plate consists entirely of '#'"

    if not GOST_TARGET_REGEX.match(norm):
        return False, f"Plate '{norm}' does not match GOST mask [L][DDD][LL][RR(R)]"

    region = norm[6:]
    if region in ("00", "000"):
        return False, f"Invalid region code '{region}'"

    return True, "OK"


def resolve_plate_type_and_validate(
    raw_plate_num: str,
    predicted_type: str,
    max_hashes: int = 4
) -> Tuple[str, str]:
    norm_num = normalize_plate_string(raw_plate_num)
    if predicted_type not in VALID_PLATE_TYPES:
        predicted_type = "other"

    if predicted_type == "other":
        return norm_num, "other"

    is_valid, _ = is_valid_gost_target_mask(norm_num, max_hashes=max_hashes)
    if not is_valid:
        return norm_num, "other"

    return norm_num, predicted_type
