import os
import cv2
import glob
from anpr_utils import get_yolo_model, preprocess_for_ocr, run_paddle_ocr, is_valid_plate_text

# Get latest uploaded file
upload_folder = "static/uploads"
files = glob.glob(os.path.join(upload_folder, "*"))

if files:
    latest_file = max(files, key=os.path.getctime)
    print(f"[INPUT] {latest_file}")
    
    img = cv2.imread(latest_file)
    print(f"[IMAGE] Original shape: {img.shape}")
    
    # YOLO detection
    yolo = get_yolo_model()
    results = yolo(img)
    
    for res in results:
        for i, box in enumerate(res.boxes):
            conf = box.conf[0].item()
            if conf >= 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                print(f"[YOLO] Box {i}: conf={conf:.4f}, coords=({x1},{y1}) to ({x2},{y2}), size={x2-x1}x{y2-y1}")
                
                # Extract ROI
                plate_roi = img[y1:y2, x1:x2]
                print(f"[ROI] Shape: {plate_roi.shape}")
                
                # Save original ROI
                cv2.imwrite(f"diagnostic_roi_original_{i}.jpg", plate_roi)
                
                # Preprocess
                preprocessed = preprocess_for_ocr(plate_roi)
                print(f"[PREPROCESSED] Shape: {preprocessed.shape}")
                
                # Save preprocessed
                cv2.imwrite(f"diagnostic_roi_preprocessed_{i}.jpg", preprocessed)
                
                # OCR
                text = run_paddle_ocr(preprocessed)
                print(f"[OCR] Raw text: {text}")
                
                if text:
                    is_valid = is_valid_plate_text(text)
                    print(f"[VALID] {is_valid}")
                    if not is_valid:
                        # Analyze why
                        print(f"  Length: {len(text)} (needs 5-20)")
                        letters = sum(1 for c in text if c.isalpha())
                        digits = sum(1 for c in text if c.isdigit())
                        alphanumeric = sum(1 for c in text if c.isalnum())
                        print(f"  Letters: {letters}, Digits: {digits}, Total alphanumeric: {alphanumeric}/{len(text)}")
                
                print(f"\n[SAVED] diagnostic_roi_original_{i}.jpg and diagnostic_roi_preprocessed_{i}.jpg")
else:
    print("[ERROR] No files in uploads")
