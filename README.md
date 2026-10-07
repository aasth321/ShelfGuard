# ShelfGuard-AI

AI-powered shelf monitoring system that uses computer vision to detect products, compare shelf placement with a planogram, and generate alerts for stockouts, low-stock items, and misplaced products.

## Features

* Product detection using YOLO
* Planogram-based shelf validation
* Stockout and low-stock detection
* Misplaced product detection
* Automatic alerts
* Database storage
* Web-based monitoring dashboard

## Project Structure

```text
ShelfGuard-AI/
│
├── backend/
│   ├── app.py
│   ├── detector.py
│   ├── planogram.py
│   ├── alerts.py
│   └── database.py
│
├── models/
│   └── product_detection.pt
│
├── data/
│   ├── shelf_images/
│   └── planograms/
│
├── frontend/
│   ├── index.html
│   ├── dashboard.js
│   └── style.css
│
├── training/
│   ├── dataset.yaml
│   └── train.py
│
├── requirements.txt
├── .env.example
├── REQUIREMENTS.md
├── ARCHITECTURE.md
└── README.md
```

## Installation

```bash
git clone <repository-url>
cd ShelfGuard-AI

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

## Configuration

Create a `.env` file using `.env.example` and add the required configuration.

Place the trained
