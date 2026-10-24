"""
Модуль формирования выходного CSV-файла в строгом соответствии с разделом 5 ТЗ.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.gost_rules import VALID_PLATE_TYPES, normalize_plate_string


@dataclass
class PlatePrediction:
    image_name: str
    plate_num: str
    plate_type: str
    confidence: float

    def validate_and_format(self) -> tuple[str, str, str, str]:
        clean_image_name = Path(self.image_name).name
        clean_plate_num = normalize_plate_string(self.plate_num)

        if self.plate_type not in VALID_PLATE_TYPES:
            raise ValueError(f"Invalid plate_type '{self.plate_type}'. Expected one of {VALID_PLATE_TYPES}")

        clamped_conf = max(0.0, min(1.0, float(self.confidence)))
        conf_str = f"{clamped_conf:.2f}"
        return clean_image_name, clean_plate_num, self.plate_type, conf_str


def write_predictions_csv(predictions: Iterable[PlatePrediction], output_path: str | Path) -> None:
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";", lineterminator="\n")
        writer.writerow(["image", "plate_num", "plate_type", "confidence"])

        for pred in predictions:
            row = pred.validate_and_format()
            writer.writerow(row)
