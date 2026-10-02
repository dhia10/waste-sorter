"""
Unit Tests for Waste Classification Logic
"""

import unittest
import numpy as np
from src.classifier import WasteClassifier, WasteCategory
from main import generate_synthetic_frame


class TestWasteClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = WasteClassifier(min_contour_area=1000)

    def test_organic_detection(self):
        frame = generate_synthetic_frame("ORGANIC")
        result = self.classifier.process_frame(frame)
        self.assertTrue(result["detected"])
        self.assertEqual(result["category"], WasteCategory.ORGANIC)
        self.assertGreater(result["confidence"], 0.70)

    def test_metal_detection(self):
        frame = generate_synthetic_frame("METAL")
        result = self.classifier.process_frame(frame)
        self.assertTrue(result["detected"])
        self.assertEqual(result["category"], WasteCategory.METAL)
        self.assertGreater(result["confidence"], 0.70)

    def test_plastic_detection(self):
        frame = generate_synthetic_frame("PLASTIC")
        result = self.classifier.process_frame(frame)
        self.assertTrue(result["detected"])
        self.assertEqual(result["category"], WasteCategory.PLASTIC)
        self.assertGreater(result["confidence"], 0.65)

    def test_empty_frame_handling(self):
        empty_frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        result = self.classifier.process_frame(empty_frame)
        self.assertFalse(result["detected"])
        self.assertEqual(result["category"], WasteCategory.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
