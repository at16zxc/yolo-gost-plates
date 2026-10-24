"""
Внутренний валидатор датасета (по требованиям стр. 6-8 ТЗ).
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from src.gost_rules import VALID_PLATE_TYPES, normalize_plate_string

REQUIRED_META_COLUMNS = [
    "image", "plate_num", "plate_type", "bbox", "quad",
    "is_vehicle", "is_synthetic", "source", "license", "conditions"
]

ALLOWED_CONDITIONS = {
    "day", "night", "rain", "snow", "dirt", "glare", "motion_blur", "angle"
}


def validate_dataset_structure(dataset_root: str | Path) -> Dict[str, object]:
    root = Path(dataset_root)
    errors: List[str] = []
    warnings: List[str] = []

    required_paths = [
        root / "images" / "real",
        root / "images" / "synthetic",
        root / "labels",
        root / "generator",
        root / "meta.csv",
        root / "README.md",
        root / "LICENSE",
    ]
    for p in required_paths:
        if not p.exists():
            errors.append(f"Missing required path: {p.relative_to(root) if p != root else p}")

    stats = {
        "real_images_by_type": defaultdict(set),
        "real_unique_plates_by_type": defaultdict(set),
        "synthetic_images": set(),
        "total_rows": 0,
    }

    meta_path = root / "meta.csv"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            if reader.fieldnames != REQUIRED_META_COLUMNS:
                errors.append(
                    f"Invalid meta.csv header: {reader.fieldnames}. Expected: {REQUIRED_META_COLUMNS}"
                )
            else:
                for idx, row in enumerate(reader, start=2):
                    stats["total_rows"] += 1
                    img_rel = row["image"]
                    p_type = row["plate_type"]
                    p_num = row["plate_num"]

                    if p_type not in VALID_PLATE_TYPES:
                        errors.append(f"Row {idx}: invalid plate_type '{p_type}'")

                    if p_num != normalize_plate_string(p_num):
                        errors.append(f"Row {idx}: plate_num '{p_num}' contains non-Latin or lowercase chars")

                    bbox_parts = row["bbox"].split(",")
                    if len(bbox_parts) != 4:
                        errors.append(f"Row {idx}: bbox must have 4 values (x,y,w,h)")

                    quad_parts = row["quad"].split(",")
                    if len(quad_parts) != 8:
                        errors.append(f"Row {idx}: quad must have 8 values")

                    if row["is_vehicle"] not in ("0", "1"):
                        errors.append(f"Row {idx}: is_vehicle must be 0 or 1")

                    if row["is_synthetic"] not in ("0", "1"):
                        errors.append(f"Row {idx}: is_synthetic must be 0 or 1")

                    if row["is_synthetic"] == "1":
                        stats["synthetic_images"].add(img_rel)
                    else:
                        stats["real_images_by_type"][p_type].add(img_rel)
                        if p_num and "#" not in p_num:
                            stats["real_unique_plates_by_type"][p_type].add(p_num)

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "total_annotations": stats["total_rows"],
            "synthetic_images": len(stats["synthetic_images"]),
        },
    }
