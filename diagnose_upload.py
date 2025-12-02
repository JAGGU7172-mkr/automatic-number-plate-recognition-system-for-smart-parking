import os
import cv2
import logging
from anpr_utils import get_yolo_model, preprocess_for_ocr, run_paddle_ocr, detect_license_plate, is_valid_plate_text
import glob

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

# Check latest uploaded file
upload_folder = "static/uploads"
files = glob.glob(os.path.join(upload_folder, "*"))
if files:
    latest_file = max(files, key=os.path.getctime)
    print(f"\n[LATEST FILE] {latest_file}")
    print(f"[FILE SIZE] {os.path.getsize(latest_file)} bytes")
    
    # Read image
    img = cv2.imread(latest_file)
    if img is None:
        print("[ERROR] Failed to read image!")
    else:
        print(f"[IMAGE] Shape: {img.shape}, Type: {type(img)}")
        
        # Try YOLO detection
        yolo = get_yolo_model()
        print(f"[YOLO] Model loaded: {yolo is not None}")
        
        if yolo:
            results = yolo(img)
            print(f"[YOLO] Results: {len(results)} result(s)")
            
            detected_any = False
            for res in results:
                print(f"[YOLO] Boxes count: {len(res.boxes)}")
                for i, box in enumerate(res.boxes):
                    conf = box.conf[0].item()
                    print(f"[YOLO] Box {i}: confidence={conf:.4f}")
                    if conf >= 0.5:
                        detected_any = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        print(f"  Coordinates: ({x1}, {y1}) to ({x2}, {y2})")
                        
                        # Try OCR
                        plate_roi = img[y1:y2, x1:x2]
                        print(f"  ROI shape: {plate_roi.shape}")
                        
                        if plate_roi.size > 0:
                            prep_roi = preprocess_for_ocr(plate_roi)
                            text = run_paddle_ocr(prep_roi)
                            print(f"  OCR result: {text}")
                            
                            if text:
                                is_valid = is_valid_plate_text(text)
                                print(f"  Validation: {is_valid}")
                                if not is_valid:
                                    print(f"    - Text length: {len(text)} (valid: 6-15)")
                                    letters = sum(1 for c in text if c.isalpha())
                                    digits = sum(1 for c in text if c.isdigit())
                                    print(f"    - Letters: {letters}, Digits: {digits}")
            
            if not detected_any:
                print("[WARNING] YOLO found boxes but none with confidence >= 0.5")
else:
    print("[ERROR] No files in uploads folder")
