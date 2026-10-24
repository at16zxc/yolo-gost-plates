"""
Модуль геометрической коррекции номерных знаков по 4 угловым точкам (quad).
"""

import cv2
import numpy as np


def order_quad_points_clockwise(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def rectify_plate(
    image: np.ndarray,
    quad: np.ndarray,
    plate_type: str,
    target_width: int = 256,
    target_height: int = 64
) -> np.ndarray:
    ordered_quad = order_quad_points_clockwise(quad)

    if plate_type == "type1a":
        sq_w, sq_h = 192, 112
        dst_pts = np.array([
            [0, 0],
            [sq_w - 1, 0],
            [sq_w - 1, sq_h - 1],
            [0, sq_h - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(ordered_quad, dst_pts)
        warped_sq = cv2.warpPerspective(image, matrix, (sq_w, sq_h), flags=cv2.INTER_LINEAR)

        mid_y = sq_h // 2
        top_row = warped_sq[0:mid_y + 4, :]
        bottom_row = warped_sq[mid_y - 4:sq_h, :]

        half_w = target_width // 2
        top_resized = cv2.resize(top_row, (half_w, target_height), interpolation=cv2.INTER_LINEAR)
        bot_resized = cv2.resize(bottom_row, (target_width - half_w, target_height), interpolation=cv2.INTER_LINEAR)

        return np.hstack([top_resized, bot_resized])

    dst_pts = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(ordered_quad, dst_pts)
    return cv2.warpPerspective(image, matrix, (target_width, target_height), flags=cv2.INTER_LINEAR)
