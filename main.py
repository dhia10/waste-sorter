"""
Autonomous Waste Sorter — Main Execution Loop
Clean & Green Hackathon (4th Place & Honorary Award)
Author: Dhia Romdhane
"""

import sys
import time
import argparse
import logging
import cv2
import numpy as np
from src.classifier import WasteClassifier
from src.actuator import ActuatorDispatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def generate_synthetic_frame(category: str) -> np.ndarray:
    """Generates synthetic test conveyor frames for headless environments or verification."""
    # 480x640 conveyor belt background (dark gray)
    frame = np.full((480, 640, 3), 45, dtype=np.uint8)

    # Belt guide lines
    cv2.line(frame, (80, 0), (80, 480), (70, 70, 70), 2)
    cv2.line(frame, (560, 0), (560, 480), (70, 70, 70), 2)

    center = (320, 240)
    if category == "ORGANIC":
        # Greenish/brownish organic apple/leaf shape
        cv2.circle(frame, center, 70, (25, 120, 40), -1)
        cv2.circle(frame, (340, 220), 40, (30, 90, 160), -1)
    elif category == "PLASTIC":
        # Translucent / bright blue plastic bottle shape
        cv2.rectangle(frame, (270, 160), (370, 320), (220, 140, 30), -1)
        cv2.rectangle(frame, (300, 130), (340, 160), (200, 200, 200), -1)
    elif category == "METAL":
        # Shiny aluminum can with high specular highlights
        cv2.rectangle(frame, (280, 170), (360, 310), (190, 190, 190), -1)
        cv2.circle(frame, (305, 210), 25, (255, 255, 255), -1)  # Specular highlight
    return frame


def run_pipeline(source: str = "demo", headless: bool = True):
    classifier = WasteClassifier(min_contour_area=1200)
    actuator = ActuatorDispatcher()

    logging.info(f"Starting Waste Sorter Pipeline (Source: {source}, Headless: {headless})")

    if source == "demo":
        categories = ["ORGANIC", "PLASTIC", "METAL"]
        for cat in categories:
            frame = generate_synthetic_frame(cat)
            start_t = time.perf_counter()
            result = classifier.process_frame(frame)
            latency_ms = (time.perf_counter() - start_t) * 1000

            if result["detected"]:
                actuator.trigger_sorting(result["category"], result["confidence"])
                logging.info(
                    f"Sample [{cat}] -> Predicted: {result['category']} "
                    f"| Confidence: {result['confidence']:.2f} "
                    f"| Latency: {latency_ms:.1f}ms "
                    f"| Features: {result['features']}"
                )
            time.sleep(0.1)
        return

    # Video stream or webcam capture
    cap_id = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_id)
    if not cap.isOpened():
        logging.error(f"Cannot open video source: {source}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_t = time.perf_counter()
        result = classifier.process_frame(frame)
        latency_ms = (time.perf_counter() - start_t) * 1000

        if result["detected"]:
            actuator.trigger_sorting(result["category"], result["confidence"])
            x, y, w, h = result["bbox"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{result['category']} ({result['confidence']*100:.0f}%) - {latency_ms:.1f}ms",
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        if not headless:
            cv2.imshow("Waste Sorting Conveyor — Live Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Waste Sorter CV Pipeline")
    parser.add_argument("--source", default="demo", help="'demo', camera index (e.g. 0), or path to video file")
    parser.add_argument("--gui", action="store_true", help="Display OpenCV GUI window")
    args = parser.parse_args()

    run_pipeline(source=args.source, headless=not args.gui)
