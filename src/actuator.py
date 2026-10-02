"""
Actuator Dispatcher (Hardware Handshake)
Clean & Green Hackathon — Sousse / ESPRIT
Author: Dhia Romdhane
"""

import time
import json
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ActuatorDispatcher:
    """
    Communicates classification results to physical sorting mechanisms (solenoids, servos).
    Supports HTTP Webhook endpoint or simulated Serial fallback.
    """

    GATE_MAPPING = {
        "ORGANIC": 1,    # Chute 1
        "PLASTIC": 2,    # Chute 2
        "METAL": 3       # Chute 3
    }

    def __init__(self, endpoint_url: Optional[str] = None, serial_port: Optional[str] = None):
        self.endpoint_url = endpoint_url
        self.serial_port = serial_port
        self.last_dispatch_time = 0.0
        self.cooldown_seconds = 1.0  # Prevent spamming triggers while object traverses belt

    def trigger_sorting(self, category: str, confidence: float) -> bool:
        """Triggers the mechanical flap corresponding to the waste category."""
        now = time.time()
        if now - self.last_dispatch_time < self.cooldown_seconds:
            return False

        gate_id = self.GATE_MAPPING.get(category, 0)
        if gate_id == 0:
            return False

        payload = {
            "timestamp": round(now, 3),
            "category": category,
            "confidence": confidence,
            "gate_id": gate_id,
            "pulse_duration_ms": 350
        }

        logging.info(f"[ACTUATOR] Triggering Gate {gate_id} ({category}) with confidence {confidence:.2f}")

        # HTTP dispatch if configured
        if self.endpoint_url:
            try:
                import requests
                resp = requests.post(self.endpoint_url, json=payload, timeout=0.5)
                self.last_dispatch_time = now
                return resp.status_code == 200
            except Exception as e:
                logging.warning(f"[ACTUATOR] HTTP trigger failed: {e}")

        # Serial dispatch fallback if configured
        if self.serial_port:
            try:
                import serial
                with serial.Serial(self.serial_port, 9600, timeout=0.1) as ser:
                    ser.write(f"GATE:{gate_id}\n".encode("utf-8"))
                self.last_dispatch_time = now
                return True
            except Exception as e:
                logging.warning(f"[ACTUATOR] Serial trigger failed: {e}")

        self.last_dispatch_time = now
        return True
