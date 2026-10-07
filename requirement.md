# Requirements

## Functional Requirements

* Detect products from shelf images using a trained YOLO model.
* Compare detected products with the store planogram.
* Identify stockouts, low-stock items, and misplaced products.
* Generate alerts for detected shelf issues.
* Store detection and alert data in a database.
* Display shelf status and alerts on a web dashboard.

## Non-Functional Requirements

* Fast product detection.
* Simple and responsive dashboard.
* Reliable database storage.
* Modular and easy-to-maintain code.
* Support for future model improvements.

## Software Requirements

* Python 3.10+
* Flask
* OpenCV
* YOLO / Ultralytics
* NumPy
* SQLite
* HTML, CSS, JavaScript

## Hardware Requirements

* Computer with camera/shelf images.
* GPU recommended for model training and faster detection.
* Sufficient storage for images and trained models.
