from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np
import cv2
from PIL import Image, ImageDraw

from src.agent.policies import Policies


Alignment = Literal["left", "center", "right"]


@dataclass(frozen=True)
class LineStyle:
    bold: bool
    italic: bool
    alignment: Alignment


@dataclass(frozen=True)
class LineBox:
    x: int
    y: int
    w: int
    h: int
    style: LineStyle


@dataclass(frozen=True)
class LayoutAnalysis:
    width: int
    height: int
    lines: List[LineBox]


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _binarize(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    return thr


def _find_line_boxes(binary_inv: np.ndarray, p: Policies) -> List[Tuple[int, int, int, int]]:
    h, w = binary_inv.shape[:2]
    kernel_w = max(p.morph_kernel_min_width, w // p.morph_kernel_width_divisor)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, p.morph_kernel_height))
    merged = cv2.dilate(binary_inv, kernel, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < w * p.contour_min_width_ratio or bh < p.contour_min_height_px:
            continue
        boxes.append((x, y, bw, bh))

    boxes.sort(key=lambda b: (b[1], b[0]))

    merged_boxes: List[Tuple[int, int, int, int]] = []
    for x, y, bw, bh in boxes:
        if not merged_boxes:
            merged_boxes.append((x, y, bw, bh))
            continue
        px, py, pw, ph = merged_boxes[-1]
        same_row = abs((y + bh // 2) - (py + ph // 2)) < max(
            p.line_merge_min_height_px, int(p.line_merge_height_ratio * max(bh, ph))
        )
        close = (x - (px + pw)) < p.line_merge_max_gap_px
        if same_row and close:
            nx = min(px, x)
            ny = min(py, y)
            nx2 = max(px + pw, x + bw)
            ny2 = max(py + ph, y + bh)
            merged_boxes[-1] = (nx, ny, nx2 - nx, ny2 - ny)
        else:
            merged_boxes.append((x, y, bw, bh))
    return merged_boxes


def _infer_alignment(box: Tuple[int, int, int, int], page_w: int, p: Policies) -> Alignment:
    x, _, bw, _ = box
    left = x / max(1, page_w)
    right = (page_w - (x + bw)) / max(1, page_w)

    if (
        abs(left - right) < p.align_center_margin_diff
        and left > p.align_center_min_margin
        and right > p.align_center_min_margin
    ):
        return "center"
    if right < p.align_right_max_right_margin and left > p.align_right_min_left_margin:
        return "right"
    return "left"


def _ink_density(binary_inv_crop: np.ndarray) -> float:
    return float(np.mean(binary_inv_crop > 0))


def _infer_bold(binary_inv_crop: np.ndarray, p: Policies) -> bool:
    d = _ink_density(binary_inv_crop)
    return d > p.bold_ink_density_threshold


def _infer_italic(gray_crop: np.ndarray, p: Policies) -> bool:
    edges = cv2.Canny(gray_crop, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=20, maxLineGap=8)
    if lines is None:
        return False
    angles = []
    for (x1, y1, x2, y2) in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        if -40 <= angle <= 40:
            angles.append(angle)
    if len(angles) < 6:
        return False
    median = float(np.median(angles))
    return p.italic_min_angle_deg <= abs(median) <= p.italic_max_angle_deg


def analyze_layout(image: Image.Image, policies: Policies | None = None) -> LayoutAnalysis:
    p = policies or Policies()
    bgr = _pil_to_bgr(image)
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bin_inv = _binarize(gray)

    boxes = _find_line_boxes(bin_inv, p)
    lines: List[LineBox] = []
    for (x, y, bw, bh) in boxes:
        crop_bin = bin_inv[max(0, y) : min(h, y + bh), max(0, x) : min(w, x + bw)]
        crop_gray = gray[max(0, y) : min(h, y + bh), max(0, x) : min(w, x + bw)]

        alignment = _infer_alignment((x, y, bw, bh), page_w=w, p=p)
        bold = _infer_bold(crop_bin, p)
        italic = _infer_italic(crop_gray, p)

        lines.append(
            LineBox(
                x=int(x),
                y=int(y),
                w=int(bw),
                h=int(bh),
                style=LineStyle(bold=bold, italic=italic, alignment=alignment),
            )
        )

    return LayoutAnalysis(width=int(w), height=int(h), lines=lines)


def render_layout_debug(image: Image.Image, layout: LayoutAnalysis) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    for line in layout.lines:
        x1, y1 = line.x, line.y
        x2, y2 = line.x + line.w, line.y + line.h
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        label = f"{line.style.alignment} | {'B' if line.style.bold else '-'}{'I' if line.style.italic else '-'}"
        draw.text((x1, max(0, y1 - 14)), label, fill=(255, 0, 0))
    return out
