"""Shared helpers for all phases."""
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.resolve()
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def class_names():
    return load_config()["classes"]


def list_images(folder):
    return sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in IMG_EXT)
