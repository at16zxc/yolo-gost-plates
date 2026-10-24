"""
Сквозной пайплайн инференса (Detector -> Rectifier -> OCR / Validator -> CSV Writer).
"""

import time
from pathlib import Path
import cv2
import numpy as np

from src.csv_writer import PlatePrediction
from src.detector import PlateDetector
from src.gost_rules import resolve_plate_type_and_validate
from src.rectifiers import rectify_plate


class RecognitionPipeline:
    def __init__(self, config: dict):
        models_cfg = config.get("models", {})
        thresholds = config.get("thresholds", {})
        
        det_path = models_cfg.get("detector_path", "./models/detector_quad.onnx")
        det_conf = thresholds.get("det_conf", 0.35)
        
        self.detector = PlateDetector(model_path=det_path, conf_threshold=det_conf)
        self.max_hashes = thresholds.get("max_hash_count", 4)
        self.emit_other = config.get("postprocessing", {}).get("emit_other_class", False)

    def process_image(self, img_path: Path) -> list[PlatePrediction]:
        # Безопасное чтение изображений с поддержкой кириллицы в путях Windows
        img_bytes = np.fromfile(str(img_path), dtype=np.uint8)
        image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return []

        detections = self.detector.detect(image)
        predictions = []

        for det in detections:
            quad = det["quad"]
            p_type = det["plate_type"]
            det_conf = det["confidence"]

            # Геометрическое выпрямление (включая склейку двух строк type1a)
            rectified_strip = rectify_plate(image, quad, plate_type=p_type)

            # На этапе baseline (до подключения весов OCR) используем заглушку чтения символов
            raw_text = "A123BC777" if p_type != "other" else "7777AB77"
            
            # Валидация по правилам ГОСТ и коррекция типа знака
            clean_num, validated_type = resolve_plate_type_and_validate(raw_text, p_type, max_hashes=self.max_hashes)

            if validated_type == "other" and not self.emit_other:
                continue  # Пропускаем 'other', если в конфиге выключен их вывод (стр. 3 ТЗ)

            predictions.append(
                PlatePrediction(
                    image_name=img_path.name,
                    plate_num=clean_num,
                    plate_type=validated_type,
                    confidence=float(det_conf)
                )
            )

        return predictions