"""
Модуль детекции номерных знаков и ключевых точек (quad) на базе ONNX Runtime.
Поддерживает автоматический выбор аппаратного ускорения: DmlExecutionProvider (AMD RX 6600),
CUDAExecutionProvider (NVIDIA GTX 1050 Ti) и CPU fallback.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import onnxruntime as ort


class PlateDetector:
    def __init__(self, model_path: str | Path, conf_threshold: float = 0.35, input_size: Tuple[int, int] = (640, 640)):
        self.conf_threshold = conf_threshold
        self.input_size = input_size
        self.model_path = Path(model_path)

        available_providers = ort.get_available_providers()
        preferred_providers = []

        if "CUDAExecutionProvider" in available_providers:
            preferred_providers.append("CUDAExecutionProvider")
        if "DmlExecutionProvider" in available_providers:
            preferred_providers.append("DmlExecutionProvider")
        preferred_providers.append("CPUExecutionProvider")

        print(f"[INFO] Инициализация PlateDetector. Доступные провайдеры ONNX: {preferred_providers}")
        
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1

        if self.model_path.exists():
            self.session = ort.InferenceSession(str(self.model_path), opts, providers=preferred_providers)
            self.is_dummy = False
        else:
            print(f"[WARN] Веса модели не найдены по пути {self.model_path}. Включен режим Baseline Mock.")
            self.is_dummy = True

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int], int, int]:
        h, w = image.shape[:2]
        target_w, target_h = self.input_size

        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = resized

        blob = canvas.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        return blob, scale, (w, h), (target_w - nw) // 2, (target_h - nh) // 2

    def detect(self, image: np.ndarray) -> List[Dict[str, np.ndarray]]:
        h_orig, w_orig = image.shape[:2]

        if self.is_dummy:
            bx, by, bw, bh = int(w_orig * 0.3), int(h_orig * 0.4), int(w_orig * 0.4), int(h_orig * 0.2)
            quad = np.array([[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]], dtype=np.float32)
            return [{"bbox": np.array([bx, by, bw, bh]), "quad": quad, "plate_type": "type1", "confidence": 0.85}]

        blob, scale, (w_orig, h_orig), pad_x, pad_y = self.preprocess(image)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: blob})
        
        preds = outputs[0]  # Формат YOLOv8: [1, 84, 8400]
        preds = np.transpose(preds, (0, 2, 1))[0]  # [8400, 84]

        class_names = ["type1", "type1a", "type1b", "other"]
        results = []

        for row in preds:
            box = row[:4]
            classes_scores = row[4:]
            class_id = np.argmax(classes_scores)
            confidence = float(classes_scores[class_id])

            if confidence < self.conf_threshold:
                continue

            cx, cy, bw, bh = box
            # Перевод координат из формата YOLO в оригинальное изображение
            x = (cx - bw / 2 - pad_x) / scale
            y = (cy - bh / 2 - pad_y) / scale
            w = bw / scale
            h = bh / scale

            # Ограничение рамки в пределах кадра
            x = max(0, min(x, w_orig))
            y = max(0, min(y, h_orig))
            w = min(w, w_orig - x)
            h = min(h, h_orig - y)

            if w <= 0 or h <= 0:
                continue

            quad = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ], dtype=np.float32)

            results.append({
                "bbox": np.array([x, y, w, h], dtype=np.float32),
                "quad": quad,
                "plate_type": class_names[class_id] if class_id < len(class_names) else "type1",
                "confidence": confidence
            })

        # Non-Maximum Suppression (NMS) для удаления пересекающихся рамок
        if len(results) > 1:
            boxes = [r["bbox"] for r in results]
            scores = [r["confidence"] for r in results]
            indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=self.conf_threshold, nms_threshold=0.45)
            if len(indices) > 0:
                results = [results[i] for i in indices.flatten()]

        return results