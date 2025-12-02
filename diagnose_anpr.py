#!/usr/bin/env python
"""
ANPR Diagnostic Test - Verify License Plate Detection Pipeline
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Test setup
print("[*] ANPR Pipeline Diagnostic Test")
print("=" * 70)

# Test 1: Check model file exists
print("\n[1] Checking YOLO Model File...")
model_path = os.path.join("static", "models", "licence_plate.pt")
alt_path = os.path.join("models", "licence_plate.pt")

if os.path.exists(model_path):
    print(f"  [OK] Model found at: {model_path}")
    print(f"  [OK] File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
elif os.path.exists(alt_path):
    print(f"  [OK] Model found at: {alt_path}")
    print(f"  [OK] File size: {os.path.getsize(alt_path) / (1024*1024):.2f} MB")
else:
    print(f"  [ERROR] Model not found at either location!")
    sys.exit(1)

# Test 2: Load YOLO Model
print("\n[2] Loading YOLO Model...")
try:
    from ultralytics import YOLO
    yolo = YOLO(model_path if os.path.exists(model_path) else alt_path)
    print("  [OK] YOLO loaded successfully")
except Exception as e:
    print(f"  [ERROR] Failed to load YOLO: {e}")
    sys.exit(1)

# Test 3: Load PaddleOCR (via shared utility so the behavior matches the app)
print("\n[3] Loading PaddleOCR...")
try:
    from anpr_utils import get_paddle_ocr

    ocr = get_paddle_ocr()
    if ocr is None:
        print("  [ERROR] Failed to load PaddleOCR (check that the 'paddle' / 'paddlepaddle' package is installed)")
        sys.exit(1)
    print("  [OK] PaddleOCR loaded successfully")
except Exception as e:
    print(f"  [ERROR] Failed to load PaddleOCR: {e}")
    sys.exit(1)

# Test 4: Test with a sample image (if exists)
print("\n[4] Testing with Sample Images...")
sample_dirs = [
    "static/uploads",
    "uploads",
]

sample_images = []
for dir_path in sample_dirs:
    if os.path.exists(dir_path):
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    sample_images.append(os.path.join(root, file))

if sample_images:
    print(f"  [OK] Found {len(sample_images)} image(s)")
    
    for img_path in sample_images[:1]:  # Test first image only
        print(f"\n  Testing: {img_path}")
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"    [ERROR] Failed to read image")
            continue
        
        print(f"    [OK] Image shape: {image.shape}")
        
        # Run YOLO
        print(f"    [*] Running YOLO detection...")
        results = yolo(image)
        
        detected_count = 0
        for res in results:
            for box in res.boxes:
                detected_count += 1
                conf = box.conf[0].item()
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                print(f"    [OK] Detection #{detected_count}")
                print(f"        - Confidence: {conf:.2%}")
                print(f"        - Box: ({x1},{y1}) to ({x2},{y2})")
                print(f"        - Size: {x2-x1}x{y2-y1} pixels")
                
                # Extract and preprocess ROI
                plate_roi = image[y1:y2, x1:x2]
                print(f"        - ROI shape: {plate_roi.shape}")
                print(f"        - ROI size: {plate_roi.size}")
                
                if plate_roi.size == 0:
                    print(f"        [WARN] ROI is empty")
                    continue
                
                # Preprocess
                gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                preprocessed = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
                print(f"        [OK] Image preprocessed")
                
                # Run OCR using the shared parsing logic from anpr_utils
                print(f"        [*] Running PaddleOCR via run_paddle_ocr...")
                try:
                    from anpr_utils import run_paddle_ocr

                    plate_text = run_paddle_ocr(preprocessed)
                    if plate_text:
                        print(f"        [OK] Extracted text: '{plate_text}'")
                    else:
                        print("        [WARN] No OCR text extracted")
                except Exception as e:
                    print(f"        [ERROR] OCR call failed: {e}")
        
        if detected_count == 0:
            print(f"    [WARN] No plate detected in image")
else:
    print("  [WARN] No sample images found to test")

# Test 5: Test OCR directly on a test image
print("\n[5] Testing OCR with Direct Text...")
try:
    # Create a test image with simple text
    test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "ABC123", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    
    print("  [*] Created test image with 'ABC123'")
    
    ocr_result = ocr.ocr(test_image)
    print(f"  [OK] OCR result: {ocr_result}")
except Exception as e:
    print(f"  [ERROR] OCR test failed: {e}")

print("\n" + "=" * 70)
print("[OK] Diagnostic test complete!")
