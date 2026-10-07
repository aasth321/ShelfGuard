# ShelfGuard-AI – Architecture

## System Flow

```text
Shelf Camera / Images
        ↓
   Product Detector
        ↓
  Detected Products
        ↓
    Planogram Check
        ↓
 ┌────────┬──────────┐
 ↓        ↓          ↓
Stockout  Misplaced  Low Stock
 └────────┴──────────┘
        ↓
      Alerts
        ↓
    Database
        ↓
    Dashboard
```

## Core Components

* **Detector (`detector.py`)**
  Detects products from shelf images using the trained YOLO model.

* **Planogram (`planogram.py`)**
  Compares detected products with the expected shelf arrangement.

* **Alerts (`alerts.py`)**
  Generates alerts for stockouts, low-stock products, and misplaced items.

* **Database (`database.py`)**
  Stores product detections, shelf status, and alerts.

* **Backend (`app.py`)**
  Connects detection, planogram checking, alerts, database, and frontend.

* **Frontend**
  Displays shelf status, detected issues, and alerts.

* **Training**
  Contains the dataset configuration and model-training script.

## Data Flow

```text
Image → YOLO Detection → Planogram Validation
      → Issue Detection → Database → Dashboard
```
