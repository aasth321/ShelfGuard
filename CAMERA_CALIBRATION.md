# Fixed camera calibration (synthetic dataset)
Image size        : 960 x 720 px
Shelves per image : 4 (top to bottom)
Slots per shelf   : 8
Shelf x-edges     : left = 40 px, right = 920 px  (110 px per slot)
Shelf bottoms (y) : ~180, 360, 540, 700 px

Phase 7 command:
  python phase7_shelf_mapping.py --source <image> --slots 8 --x-range 40 920

Folders
  data/raw/images + labels      400 training images with YOLO labels
  data/heldout/images + labels   30 test images, NEVER used in training
