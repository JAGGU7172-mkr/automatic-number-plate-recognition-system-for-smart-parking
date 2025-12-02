#!/usr/bin/env python
"""
Test ANPR OCR Text Extraction - Verify Fixed Parsing
"""
import os
import sys
import cv2

print("[*] Testing ANPR OCR Text Extraction")
print("=" * 70)

# Import the fixed OCR function
from anpr_utils import get_paddle_ocr, preprocess_for_ocr, get_yolo_model, detect_license_plate
import numpy as np

# Test 1: Direct OCR test with known text
print("\n[1] Testing OCR with Test Image...")
try:
    ocr = get_paddle_ocr()
    if not ocr:
        print("  [ERROR] Failed to load PaddleOCR")
        sys.exit(1)
    
    # Create test image with text
    test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "ABC123", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    
    result = ocr.ocr(test_image)
    print(f"  [OK] OCR returned result")
    
    if result and len(result) > 0 and isinstance(result[0], dict):
        if 'rec_texts' in result[0]:
            texts = result[0]['rec_texts']
            scores = result[0]['rec_scores']
            print(f"  [OK] Format: NEW (rec_texts/rec_scores)")
            print(f"  [OK] Detected texts: {texts}")
            print(f"  [OK] Scores: {scores}")
            
            if 'ABC123' in texts or 'ABC' in str(texts):
                print(f"  [SUCCESS] Test text detected!")
            else:
                print(f"  [WARN] Test text not found in results")
    else:
        print(f"  [WARN] Unexpected result format: {type(result)}")
        
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()

# Test 2: Full pipeline test on real image
print("\n[2] Testing Full ANPR Pipeline...")
sample_dirs = ["static/uploads", "uploads"]
sample_found = False

for dir_path in sample_dirs:
    if os.path.exists(dir_path):
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, file)
                    print(f"\n  Testing: {img_path}")
                    
                    try:
                        plate, conf, proc_path = detect_license_plate(img_path)
                        
                        if plate:
                            print(f"  [SUCCESS] Plate detected: {plate}")
                            print(f"  [OK] Confidence: {conf:.2%}")
                            print(f"  [OK] Processed image: {proc_path}")
                        else:
                            print(f"  [INFO] No plate detected (may be image issue)")
                        
                        sample_found = True
                        break
                    except Exception as e:
                        print(f"  [ERROR] {e}")
                        import traceback
                        traceback.print_exc()
                    
                    if sample_found:
                        break
            if sample_found:
                break
    if sample_found:
        break

if not sample_found:
    print("  [WARN] No sample images found")

print("\n" + "=" * 70)
print("[OK] OCR Testing Complete!")
