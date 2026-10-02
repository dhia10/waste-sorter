# Autonomous Waste Sorter (Computer Vision & Edge Sorting)

Real-time computer vision pipeline developed in a 48-hour collaborative sprint during the **Clean & Green Hackathon** (awarded **4th Place & Honorary Prize**).

The system classifies solid waste moving along a conveyor belt into three discrete streams:
1. **ORGANIC** (compostable matter, food scrap profiles)
2. **PLASTIC** (PET bottles, HDPE packaging, translucent polymer profiles)
3. **METAL** (aluminum cans, cutlery, high specular glare profiles)

Once an item is categorized above the confidence threshold, the pipeline sends a sorting command to physical actuators (servo/solenoid gates) via HTTP or Serial interface.

---

## Architecture & Detection Pipeline

```
[ Camera Stream / Video Feed ]
              │
              ▼
   [ Gaussian Blur & Canny ]  ──> Contour extraction & ROI bounding box
              │
              ▼
    [ Feature Extraction ]    ──> Specular ratio, HSV distribution, Laplacian variance
              │
              ▼
    [ Threshold Classifier ]  ──> Category (ORGANIC / PLASTIC / METAL) + Confidence
              │
              ▼
   [ Actuator Dispatcher ]   ──> Cooldown gate trigger (HTTP POST / Serial)
```

### Key Technical Characteristics
- **Visual Features:** Hue/Saturation/Value (HSV) masks, specular reflection ratios (thresholded brightness for metallic glares), and Laplacian variance for surface roughness.
- **Inference Latency:** 5–15 ms per frame on standard CPU (capable of 40+ FPS sustained processing).
- **False-Positive Mitigation:** Cooldown timer (1.0s window) prevents re-triggering mechanical gates while an item traverses the camera field of view.

---

## Project Structure

```
waste-sorter/
├── src/
│   ├── classifier.py       # OpenCV contour extraction and feature classification
│   ├── actuator.py         # Hardware trigger dispatch (HTTP / Serial)
│   └── camera.py           # Video capture wrapper
├── tests/
│   └── test_classifier.py  # Unit test suite for classification and edge cases
├── main.py                 # Pipeline runner (supports --source demo, camera, or file)
├── requirements.txt        # Minimal dependencies (opencv-python, numpy, requests)
└── README.md
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/dhia10/waste-sorter.git
cd waste-sorter

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Synthetic Demo

Runs a 3-category simulated conveyor inspection cycle with latency metrics:

```bash
python main.py --source demo
```

### 3. Run on Live Webcam

```bash
python main.py --source 0 --gui
```

### 4. Run Unit Tests

```bash
python -m unittest discover -s tests
```

---

## Author & Contact

- **Developer:** Dhia Romdhane — Data Science & AI Engineering (ESPRIT)
- **LinkedIn:** [linkedin.com/in/dhia-romdhane-ds](https://www.linkedin.com/in/dhia-romdhane-ds/)
- **GitHub:** [github.com/dhia10](https://github.com/dhia10)

---

## License

MIT License.
