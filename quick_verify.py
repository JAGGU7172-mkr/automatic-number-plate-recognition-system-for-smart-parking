#!/usr/bin/env python
"""Quick OCR Fix Verification"""
from anpr_utils import run_paddle_ocr
import numpy as np
import cv2

print("\n" + "="*60)
print("Testing OCR Text Extraction Fix")
print("="*60 + "\n")

# Create test image with text
img = np.ones((100, 300, 3), dtype=np.uint8) * 255
cv2.putText(img, 'ABC123', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

print("[*] Running OCR on test image...")
result = run_paddle_ocr(img)

print(f"\nTest Input: 'ABC123'")
print(f"OCR Output: '{result}'")

if result and 'ABC' in result:
    print("\n[OK] SUCCESS - Text extracted correctly!")
else:
    print("\n[FAIL] FAILED - Text not extracted")

print("\n" + "="*60 + "\n")
