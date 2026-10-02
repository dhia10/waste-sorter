"""
Waste Classification Engine (Computer Vision & Feature Analysis)
Clean & Green Hackathon — Sousse / ESPRIT
Author: Dhia Romdhane (Lead CV / Pipeline)
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional


class WasteCategory:
    ORGANIC = "ORGANIC"
    PLASTIC = "PLASTIC"
    METAL = "METAL"
    UNKNOWN = "UNKNOWN"


class WasteClassifier:
    def __init__(self, min_contour_area: int = 1500):
        self.min_contour_area = min_contour_area

    def extract_roi(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        Locates the primary waste object in the frame using edge detection and contour analysis.
        Returns: (roi_image, (x, y, w, h)) or (None, None)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)

        # Dilate edges to close gaps
        dilated = cv2.dilate(edged, None, iterations=2)
        contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        # Take the largest contour above the threshold
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < self.min_contour_area:
            return None, None

        x, y, w, h = cv2.boundingRect(c)
        roi = frame[y:y+h, x:x+w]
        return roi, (x, y, w, h)

    def classify_roi(self, roi: np.ndarray) -> Dict[str, Any]:
        """
        Extracts visual descriptors (color histograms, specular highlights, texture/roughness)
        to categorize into ORGANIC, PLASTIC, or METAL.
        """
        if roi is None or roi.size == 0:
            return {"category": WasteCategory.UNKNOWN, "confidence": 0.0, "features": {}}

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 1. Specular Highlights & Glare (Indicates metallic or polished hard surface)
        _, highlight_mask = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
        specular_ratio = np.count_nonzero(highlight_mask) / float(gray.size)

        # 2. Color Saturation & Value
        h_channel, s_channel, v_channel = cv2.split(hsv)
        mean_saturation = np.mean(s_channel)
        mean_value = np.mean(v_channel)
        std_value = np.std(v_channel)

        # 3. Organic Color Profile (Browns, greens, oranges, earthy tones)
        # Hue ranges: 10-35 (brown/orange), 35-85 (green/yellow-green)
        organic_mask = cv2.inRange(hsv, np.array([10, 40, 30]), np.array([85, 255, 220]))
        organic_ratio = np.count_nonzero(organic_mask) / float(gray.size)

        # 4. Texture Variance (Laplacian operator for surface roughness)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        features = {
            "specular_ratio": round(float(specular_ratio), 4),
            "mean_saturation": round(float(mean_saturation), 2),
            "mean_value": round(float(mean_value), 2),
            "organic_ratio": round(float(organic_ratio), 4),
            "laplacian_var": round(float(laplacian_var), 2)
        }

        # Decision Logic:
        # High specular highlights + low saturation + sharp edges -> Metal (aluminum cans, cutlery)
        if specular_ratio > 0.08 and mean_saturation < 85:
            confidence = min(0.95, 0.65 + (specular_ratio * 3.0))
            return {"category": WasteCategory.METAL, "confidence": round(confidence, 2), "features": features}

        # High organic color profile + textured surface -> Organic (scraps, peelings, paper/cardboard)
        if organic_ratio > 0.22 and mean_saturation > 45:
            confidence = min(0.96, 0.60 + (organic_ratio * 1.5))
            return {"category": WasteCategory.ORGANIC, "confidence": round(confidence, 2), "features": features}

        # Smooth surface, moderate specular, or transparent/colored polymer -> Recyclable Plastic
        confidence = 0.82 if mean_value > 80 else 0.74
        return {"category": WasteCategory.PLASTIC, "confidence": round(confidence, 2), "features": features}

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """End-to-end frame processing with bounding box and classification."""
        roi, bbox = self.extract_roi(frame)
        if roi is None or bbox is None:
            return {"detected": False, "category": WasteCategory.UNKNOWN, "confidence": 0.0, "bbox": None, "features": {}}

        result = self.classify_roi(roi)
        return {
            "detected": True,
            "category": result["category"],
            "confidence": result["confidence"],
            "bbox": bbox,
            "features": result["features"]
        }
