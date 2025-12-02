import os
import uuid
import logging
import cv2
import numpy as np
from PIL import Image
import re
from functools import lru_cache
# ultralytics and paddleocr are imported lazily inside getter functions

# ✅ Logging Configuration
logging.basicConfig(level=logging.INFO)

# ✅ YOLO Model Path (keep path here but delay heavy loading)
YOLO_MODEL_PATH = os.path.join("static", "models", "licence_plate.pt")
# model and paddle_ocr will be lazily initialized on first use to avoid
# blocking imports and repeated downloads during reloader restarts.
model = None
paddle_ocr = None

def get_yolo_model():
    global model
    if model is None:
        try:
            # Check if model file exists
            if not os.path.exists(YOLO_MODEL_PATH):
                logging.error(f"YOLO model file not found at: {YOLO_MODEL_PATH}")
                # Try alternate path
                alt_path = os.path.join("models", "licence_plate.pt")
                if os.path.exists(alt_path):
                    logging.info(f"Found model at alternate path: {alt_path}")
                    model_path = alt_path
                else:
                    logging.error(f"Model not found at either location")
                    return None
            else:
                model_path = YOLO_MODEL_PATH
            
            logging.info(f"Loading YOLO model from: {model_path}")
            # import here to avoid heavy imports at module import time
            from ultralytics import YOLO as _YOLO
            model = _YOLO(model_path)
            logging.info("YOLO model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}", exc_info=True)
            model = None
    return model

@lru_cache(maxsize=1)
def _get_paddle_lib_paths():
    """Return Paddle's native library folders if available."""
    try:
        import paddle  # noqa: F401  (import only to locate package path)

        paddle_pkg_path = os.path.dirname(os.path.abspath(paddle.__file__))
        candidates = [
            os.path.join(paddle_pkg_path, "base"),
            os.path.join(paddle_pkg_path, "libs"),
            os.path.join(paddle_pkg_path, "libs", "third_party"),
        ]
        return [path for path in candidates if os.path.isdir(path)]
    except Exception:
        pass
    return []


def _ensure_paddle_runtime_on_path():
    """Ensure Paddle's DLL directories are appended to PATH so libpaddle can load."""
    libs_dirs = _get_paddle_lib_paths()
    if not libs_dirs:
        return
    current_path = os.environ.get("PATH", "")
    new_paths = [d for d in libs_dirs if d not in current_path]
    for path in libs_dirs:
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(path)
            except (FileNotFoundError, OSError):
                continue
    if not new_paths:
        return
    os.environ["PATH"] = ";".join(new_paths + [current_path]) if current_path else ";".join(new_paths)


def get_paddle_ocr():
    global paddle_ocr
    if paddle_ocr is None:
        try:
            _ensure_paddle_runtime_on_path()
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            logging.info("Initializing PaddleOCR...")
            # import here to avoid heavy imports at module import time
            from paddleocr import PaddleOCR as _PaddleOCR
            # Use new parameter name for newer versions (>= 2.7)
            paddle_ocr = _PaddleOCR(use_textline_orientation=True, lang='en')
            logging.info("PaddleOCR initialized")
        except Exception as e:
            logging.error(f"Failed to initialize PaddleOCR: {e}")
            paddle_ocr = None
    return paddle_ocr

# ✅ Check if file is allowed
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# ✅ Save uploaded file
def save_uploaded_file(file, upload_folder):
    try:
        if file and file.filename:
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            file_path = os.path.join(upload_folder, filename)
            os.makedirs(upload_folder, exist_ok=True)
            file.save(file_path)
            logging.info(f"File saved to: {file_path}")
            return file_path
        return None
    except Exception as e:
        logging.error(f"Error saving file: {str(e)}")
        return None

# ✅ Preprocess Image for OCR
def preprocess_for_ocr(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    
    # Thresholding to make text more distinct
    _, threshed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert back to RGB for PaddleOCR
    return cv2.cvtColor(threshed, cv2.COLOR_GRAY2RGB)


def pad_roi(roi, pad=8):
    """Add padding around ROI and clamp to image bounds."""
    if roi is None or roi.size == 0:
        return roi
    h, w = roi.shape[:2]
    pad_x = min(pad, max(1, int(w * 0.05)))
    pad_y = min(pad, max(1, int(h * 0.05)))
    padded = cv2.copyMakeBorder(roi, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
    return padded


def deskew_image(gray):
    """Estimate skew angle and rotate image to deskew. Input should be single-channel."""
    try:
        coords = cv2.findNonZero(255 - gray)
    except Exception:
        coords = None
    if coords is None or len(coords) < 10:
        return gray
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def try_ocr_on_variations(roi):
    """Try OCR on fastest candidates first, stop when valid plate found.
    
    Uses aggressive early exit: tests original -> preprocessed -> scaled variants.
    Only tests rotations/deskew if initial attempts fail, to minimize OCR calls.
    Includes safety checks to prevent crashes.
    """
    if roi is None or roi.size == 0:
        return None

    try:
        # Validate ROI dimensions
        if len(roi.shape) < 2:
            return None
        h, w = roi.shape[:2]
        if h < 5 or w < 5:
            return None
    except Exception as e:
        logging.error(f"Invalid ROI shape: {e}")
        return None

    # Fast-path candidates to try first (in order)
    fast_candidates = []
    
    # 1. Original unmodified ROI (fastest)
    fast_candidates.append(('original', roi))
    
    # 2. Preprocessed (good results, still fast)
    try:
        pre_rgb = preprocess_for_ocr(roi)
        if pre_rgb is not None and pre_rgb.size > 0:
            pre_bgr = cv2.cvtColor(pre_rgb, cv2.COLOR_RGB2BGR)
            fast_candidates.append(('preprocessed', pre_bgr))
    except Exception as e:
        logging.debug(f"Preprocessing failed: {e}")
        pass
    
    # 3. Padded original (often helps with edge text)
    try:
        padded = pad_roi(roi, pad=8)
        if padded is not None and padded.size > 0:
            fast_candidates.append(('padded', padded))
    except Exception as e:
        logging.debug(f"Padding failed: {e}")
        pass
    
    # 4. 1.5x upscale (helps small text, faster than 2x)
    try:
        h_roi, w_roi = roi.shape[:2]
        if h_roi > 0 and w_roi > 0:
            resized = cv2.resize(roi, (max(1, int(w_roi * 1.5)), max(1, int(h_roi * 1.5))), interpolation=cv2.INTER_CUBIC)
            if resized is not None and resized.size > 0:
                fast_candidates.append(('scaled_1.5x', resized))
    except Exception as e:
        logging.debug(f"Scaling failed: {e}")
        pass

    # Try fast candidates first - exit immediately if valid plate found
    best_candidate = None
    for name, img in fast_candidates:
        try:
            if img is None or img.size == 0:
                continue
                
            if len(img.shape) >= 3 and img.shape[2] == 3:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                rgb = img
        except Exception as e:
            logging.debug(f"Color conversion failed for {name}: {e}")
            continue

        try:
            text = run_paddle_ocr(rgb)
        except Exception as e:
            logging.debug(f"Error running OCR on {name}: {e}")
            text = None

        if not text:
            continue

        try:
            tnorm = text.replace(" ", "").upper()
            found = find_plate_in_text(tnorm)
            if found:
                tnorm = found

            # Valid plate found - return immediately (EARLY EXIT)
            if is_valid_plate_text(tnorm):
                return format_plate_text(tnorm)

            # Track best non-validated candidate for fallback
            if best_candidate is None or len(tnorm) > len(best_candidate):
                best_candidate = tnorm
        except Exception as e:
            logging.debug(f"Error processing OCR text: {e}")
            continue

    # Only try slower variations if fast path didn't find valid plate
    slow_candidates = []
    
    # 5. Deskewed (can be slow, skip if fast path worked well)
    if best_candidate is None or len(best_candidate) < 8:
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            desk = deskew_image(gray)
            if desk is not None and desk.size > 0:
                desk_bgr = cv2.cvtColor(desk, cv2.COLOR_GRAY2BGR)
                slow_candidates.append(('deskewed', desk_bgr))
        except Exception as e:
            logging.debug(f"Deskewing failed: {e}")
            pass
    
    # 6. Small rotations (limited to ±4 degrees, skip ±8)
    if best_candidate is None or len(best_candidate) < 8:
        try:
            h_roi, w_roi = roi.shape[:2]
            center = (w_roi // 2, h_roi // 2)
            for a in (-4, 4):  # Only ±4, not ±8
                try:
                    M = cv2.getRotationMatrix2D(center, a, 1.0)
                    rot = cv2.warpAffine(roi, M, (w_roi, h_roi), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    if rot is not None and rot.size > 0:
                        slow_candidates.append((f'rotated_{a}', rot))
                except Exception as e:
                    logging.debug(f"Rotation {a} failed: {e}")
                    continue
        except Exception as e:
            logging.debug(f"Rotation setup failed: {e}")
            pass
    
    # 7. Extract text-like subregions from very large ROIs (only if needed)
    def extract_text_like_regions(img, max_regions=2):
        """Return max 2 best candidate sub-ROIs to avoid excessive OCR calls."""
        subs = []
        try:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(g, (3, 3), 0)
            grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
            _, th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img_h, img_w = img.shape[:2]
            rects = []
            for cnt in contours:
                x, y, ww, hh = cv2.boundingRect(cnt)
                if ww < 0.3 * img_w and hh < 0.6 * img_h and ww > 30 and hh > 10:
                    area = ww * hh
                    sub = img[max(0, y):min(img_h, y+hh), max(0, x):min(img_w, x+ww)]
                    if sub.size > 0:
                        rects.append((area, sub))
            # Return only top 2 by area to limit OCR calls
            for area, sub in sorted(rects, reverse=True)[:max_regions]:
                subs.append(sub)
        except Exception as e:
            logging.debug(f"Subregion extraction failed: {e}")
            pass
        return subs

    if best_candidate is None or len(best_candidate) < 8:
        try:
            h_roi, w_roi = roi.shape[:2]
            if h_roi * w_roi > 2000 * 2000 or (h_roi > 1000 and w_roi > 1000):
                subregions = extract_text_like_regions(roi, max_regions=2)
                for i, s in enumerate(subregions):
                    if s is not None and s.size > 0:
                        slow_candidates.append((f'subregion_{i}', s))
        except Exception as e:
            logging.debug(f"Large ROI subregion processing failed: {e}")
            pass

    # Try slow candidates with limit to 4 total attempts
    for i, (name, img) in enumerate(slow_candidates):
        if i >= 4:  # Absolute limit on slow attempts
            break
        
        try:
            if img is None or img.size == 0:
                continue
                
            if len(img.shape) >= 3 and img.shape[2] == 3:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                rgb = img
        except Exception as e:
            logging.debug(f"Color conversion failed for slow candidate {name}: {e}")
            continue

        try:
            text = run_paddle_ocr(rgb)
        except Exception as e:
            logging.debug(f"Error running OCR on {name}: {e}")
            text = None

        if not text:
            continue

        try:
            tnorm = text.replace(" ", "").upper()
            found = find_plate_in_text(tnorm)
            if found:
                tnorm = found

            if is_valid_plate_text(tnorm):
                return format_plate_text(tnorm)

            if best_candidate is None or len(tnorm) > len(best_candidate):
                best_candidate = tnorm
        except Exception as e:
            logging.debug(f"Error processing slow OCR text: {e}")
            continue

    # Return best candidate if any
    try:
        if best_candidate:
            found = find_plate_in_text(best_candidate)
            if found:
                return format_plate_text(found)
            return format_plate_text(best_candidate)
    except Exception as e:
        logging.error(f"Error formatting best candidate: {e}")
        return None
    
    return None

# ✅ PaddleOCR Function
def run_paddle_ocr(image_np):
    ocr = get_paddle_ocr()
    if ocr is None:
        logging.error("PaddleOCR is not available")
        return None
    try:
        # Call OCR without passing unexpected keyword arguments to maintain
        # compatibility across paddleocr versions
        logging.debug("Running PaddleOCR on image array")
        result = ocr.ocr(image_np)
        logging.debug(f"Raw PaddleOCR result type: {type(result)}")

        plate_text = ""
        if not result:
            logging.info("PaddleOCR returned empty result")
            return None

        # NEW FORMAT: PaddleOCR 2.x+ returns dict with 'rec_texts' key
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            if 'rec_texts' in result[0]:
                # New format with rec_texts and rec_scores
                rec_texts = result[0].get('rec_texts', [])
                rec_scores = result[0].get('rec_scores', [])
                logging.debug(f"New format - rec_texts: {rec_texts}, rec_scores: {rec_scores}")
                
                for text, score in zip(rec_texts, rec_scores):
                    if score > 0.5:  # Filter by confidence
                        plate_text += str(text)
                plate_text = plate_text.replace(" ", "").strip().upper()
                logging.info(f"[PaddleOCR] Detected: {plate_text}")
                return plate_text if plate_text else None

        # OLD FORMAT FALLBACK: list of lists [[(box), (text, conf)], ...]
        try:
            for item in result:
                if isinstance(item, (list, tuple)):
                    for sub in item:
                        if isinstance(sub, (list, tuple)) and len(sub) >= 2:
                            # sub[1] might be (text, conf) or text
                            if isinstance(sub[1], (list, tuple)) and len(sub[1]) >= 1 and isinstance(sub[1][0], str):
                                plate_text += sub[1][0]
                            elif isinstance(sub[1], str):
                                plate_text += sub[1]
                        elif isinstance(sub, str):
                            plate_text += sub
                elif isinstance(item, str):
                    plate_text += item
        except Exception as e:
            logging.debug(f"Old format parsing failed: {e}")
            # Fallback: try to extract strings recursively
            def extract_strings(x):
                s = ""
                if isinstance(x, str):
                    return x
                if isinstance(x, (list, tuple)):
                    for el in x:
                        try:
                            s += extract_strings(el)
                        except Exception:
                            continue
                return s

            plate_text = extract_strings(result)

        plate_text = plate_text.replace(" ", "").strip().upper()
        logging.info(f"[PaddleOCR] Detected: {plate_text}")
        return plate_text if plate_text else None
    except Exception as e:
        logging.error(f"[PaddleOCR Error] {e}", exc_info=True)
        return None

# ✅ Validate License Plate Format
def format_plate_text(text):
    """Normalize plate text to uppercase and insert spaces as AA 00 AA 0000 when possible.

    Returns formatted string or original normalized string if formatting not possible.
    """
    if not text:
        return None
    s = re.sub(r"\s+", "", text).upper()
    # Match Indian standard: 2 letters, 2 digits, 1-2 letters, 1-4 digits
    m = re.match(r"^([A-Z]{2})(\d{2})([A-Z]{1,2})(\d{1,4})$", s)
    if m:
        g1, g2, g3, g4 = m.groups()
        return f"{g1} {g2} {g3} {g4}"
    return s


def is_valid_plate_text(text):
    """Validate the text strictly against Indian licence plate format.

    Expected pattern: AA 00 AA 0000 (spaces optional in input). Returns True
    only if the normalized text matches the pattern.
    """
    if not text:
        return False
    s = re.sub(r"\s+", "", text).upper()
    # Strict match: two letters, two digits, one or two letters, 1-4 digits
    return bool(re.match(r"^[A-Z]{2}\d{2}[A-Z]{1,2}\d{1,4}$", s))


def find_plate_in_text(text):
    """Search for an Indian-format plate inside a longer OCR string.

    Returns the matched compact (no spaces) plate or None.
    """
    if not text:
        return None
    s = re.sub(r"\s+", "", text).upper()
    # search for the pattern anywhere in the string
    m = re.search(r"[A-Z]{2}\d{2}[A-Z]{1,2}\d{1,4}", s)
    if m:
        return m.group(0)
    return None

# ✅ License Plate Detection Logic
def detect_license_plate(image_path):
    try:
        if not os.path.exists(image_path):
            logging.error(f"Image not found: {image_path}")
            return None, 0.0, None

        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to read image: {image_path}")
            return None, 0.0, None

        if image.size == 0 or len(image.shape) < 2:
            logging.error(f"Invalid image dimensions: {image.shape}")
            return None, 0.0, None

        output_image = image.copy()
        yolo = get_yolo_model()
        if yolo is None:
            logging.error("YOLO model is not available")
            return None, 0.0, None

        try:
            results = yolo(image)
        except Exception as e:
            logging.error(f"YOLO detection failed: {e}", exc_info=True)
            return None, 0.0, None

        detected_plate = None
        detected_confidence = 0.0

        found = False
        best_unvalidated = None
        
        if not results or len(results) == 0:
            logging.info("YOLO returned no results")
            return None, 0.0, None
        
        try:
            for res in results:
                if not hasattr(res, 'boxes'):
                    continue
                    
                for box in res.boxes:
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0].item()
                        logging.info(f"YOLO detection confidence: {conf}")

                        if conf < 0.5:
                            continue

                        # Validate coordinates
                        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                            logging.warning(f"Invalid box coordinates: {x1},{y1},{x2},{y2}")
                            continue

                        # Draw bounding box
                        try:
                            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        except Exception as e:
                            logging.debug(f"Failed to draw box: {e}")

                        # Extract plate region and run OCR
                        try:
                            plate_roi = image[y1:y2, x1:x2]
                        except Exception as e:
                            logging.warning(f"Failed to extract ROI: {e}")
                            continue
                        
                        # Check if ROI is valid (has non-zero width and height)
                        if plate_roi is None or plate_roi.size == 0 or plate_roi.shape[0] < 5 or plate_roi.shape[1] < 5:
                            logging.warning(f"Detected ROI too small: {plate_roi.shape if plate_roi is not None else 'None'}. Skipping OCR.")
                            continue
                        
                        # Try OCR on multiple ROI variations (scales/padding/deskew/rotations)
                        try:
                            candidate = try_ocr_on_variations(plate_roi)
                        except Exception as e:
                            logging.error(f"Error during OCR variations: {e}", exc_info=True)
                            candidate = None

                        if candidate:
                            # store best unvalidated as fallback
                            if best_unvalidated is None or len(candidate) > len(best_unvalidated):
                                best_unvalidated = candidate

                            # prefer validated plate text
                            if is_valid_plate_text(candidate):
                                detected_plate = format_plate_text(candidate)
                                detected_confidence = conf
                                try:
                                    cv2.putText(output_image, f"Plate: {detected_plate}",
                                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6, (255, 255, 0), 2)
                                except Exception as e:
                                    logging.debug(f"Failed to put text: {e}")
                                found = True
                                break
                            else:
                                # See if a valid substring can be found inside the OCR candidate
                                inner = find_plate_in_text(candidate)
                                if inner and is_valid_plate_text(inner):
                                    detected_plate = format_plate_text(inner)
                                    detected_confidence = conf
                                    try:
                                        cv2.putText(output_image, f"Plate: {detected_plate}",
                                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.6, (255, 255, 0), 2)
                                    except Exception as e:
                                        logging.debug(f"Failed to put text: {e}")
                                    found = True
                                    break
                                logging.warning(f"OCR returned text but failed validation: {candidate} (not a valid plate format)")
                    except Exception as e:
                        logging.error(f"Error processing box: {e}", exc_info=True)
                        continue
                        
                if found:
                    break
        except Exception as e:
            logging.error(f"Error iterating results: {e}", exc_info=True)
            
        # If we didn't find a validated plate, fall back to best unvalidated candidate
        if not detected_plate and best_unvalidated:
            try:
                # If we can find a valid plate substring inside best_unvalidated, use it
                inner = find_plate_in_text(best_unvalidated)
                if inner and is_valid_plate_text(inner):
                    fmt = format_plate_text(inner)
                    logging.info(f"Falling back to inner validated OCR candidate: {fmt}")
                    detected_plate = fmt
                else:
                    # Loose acceptance criteria: reasonable length and mostly alphanumeric
                    alnum = sum(1 for c in best_unvalidated if c.isalnum())
                    if len(best_unvalidated) >= 5 and alnum >= len(best_unvalidated) * 0.5:
                        logging.info(f"Falling back to unvalidated OCR candidate: {best_unvalidated}")
                        detected_plate = format_plate_text(best_unvalidated)
            except Exception as e:
                logging.error(f"Error with fallback processing: {e}", exc_info=True)

        # Save processed image
        try:
            processed_dir = os.path.join("static", "processed")
            os.makedirs(processed_dir, exist_ok=True)
            processed_image_path = os.path.join(processed_dir, f"processed_{uuid.uuid4().hex}.jpg")
            cv2.imwrite(processed_image_path, output_image)
            logging.info(f"Processed image saved: {processed_image_path}")
        except Exception as e:
            logging.error(f"Failed to save processed image: {e}", exc_info=True)
            processed_image_path = None

        return detected_plate, detected_confidence, processed_image_path

    except Exception as e:
        logging.error(f"Detection failed: {str(e)}", exc_info=True)
        return None, 0.0, None
