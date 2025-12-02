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
    """Try OCR on several transformed versions of the ROI and return the best/first valid result.

    This keeps using the same PaddleOCR instance and models; it just tries scaled, padded,
    deskewed and small-rotation variants to improve recognition on tough images.
    """
    if roi is None or roi.size == 0:
        return None

    candidates = []

    # padded original (small padding)
    try:
        candidates.append(pad_roi(roi, pad=8))
    except Exception:
        pass

    # resize variations (upsample helps OCR)
    for scale in (1.0, 1.5, 2.0):
        try:
            if scale == 1.0:
                candidates.append(roi)
            else:
                h, w = roi.shape[:2]
                resized = cv2.resize(roi, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_CUBIC)
                candidates.append(resized)
        except Exception:
            continue

    # add preprocess_for_ocr output (already returns RGB)
    try:
        pre_rgb = preprocess_for_ocr(roi)
        pre_bgr = cv2.cvtColor(pre_rgb, cv2.COLOR_RGB2BGR)
        candidates.append(pre_bgr)
    except Exception:
        pass

    # deskewed version
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        desk = deskew_image(gray)
        desk_bgr = cv2.cvtColor(desk, cv2.COLOR_GRAY2BGR)
        candidates.append(desk_bgr)
    except Exception:
        pass

    # small rotations around center
    angles = (-8, -4, 0, 4, 8)
    base_candidates = list(candidates)
    for img in base_candidates:
        if img is None:
            continue
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        for a in angles:
            if a == 0:
                continue
            try:
                M = cv2.getRotationMatrix2D(center, a, 1.0)
                rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                candidates.append(rot)
            except Exception:
                continue

    # If ROI is very large, try to find text-like subregions inside it
    def extract_text_like_regions(img):
        """Return list of candidate sub-ROIs inside a large ROI by using morphology and contours."""
        subs = []
        try:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # smooth and gradient
            blur = cv2.GaussianBlur(g, (3, 3), 0)
            grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
            _, th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # close gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            h, w = img.shape[:2]
            for cnt in contours:
                x, y, ww, hh = cv2.boundingRect(cnt)
                # plausible plate aspect ratio and size
                if ww < 0.3 * w and hh < 0.6 * h and ww > 30 and hh > 10:
                    subs.append(img[y:y+hh, x:x+ww])
        except Exception:
            pass
        return subs

    # If ROI area is very large, extract subregions and add to candidates
    try:
        h_roi, w_roi = roi.shape[:2]
        if h_roi * w_roi > 2000 * 2000 or (h_roi > 1000 and w_roi > 1000):
            subregions = extract_text_like_regions(roi)
            for s in subregions:
                candidates.append(s)
    except Exception:
        pass

    # Deduplicate by shape to reduce redundant OCR calls
    seen_shapes = set()
    final_candidates = []
    for c in candidates:
        if c is None or c.size == 0:
            continue
        key = (c.shape[0], c.shape[1])
        if key in seen_shapes:
            final_candidates.append(c)
            continue
        seen_shapes.add(key)
        final_candidates.append(c)

    best_candidate = None
    for img in final_candidates:
        try:
            # run_paddle_ocr expects RGB arrays
            if img.shape[2] == 3:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                rgb = img
        except Exception:
            rgb = img

        try:
            text = run_paddle_ocr(rgb)
        except Exception as e:
            logging.debug(f"Error running OCR on candidate: {e}")
            text = None

        if not text:
            continue

        tnorm = text.replace(" ", "").upper()
        # If a valid plate substring exists inside OCR text, prefer it
        found = find_plate_in_text(tnorm)
        if found:
            tnorm = found

        # prefer validated plate text
        if is_valid_plate_text(tnorm):
            return format_plate_text(tnorm)

        # otherwise keep longest plausible (store compact form)
        if best_candidate is None or len(tnorm) > len(best_candidate):
            best_candidate = tnorm

    # return best candidate formatted if possible
    if best_candidate:
        found = find_plate_in_text(best_candidate)
        if found:
            return format_plate_text(found)
        return format_plate_text(best_candidate)
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

        output_image = image.copy()
        yolo = get_yolo_model()
        if yolo is None:
            logging.error("YOLO model is not available")
            return None, 0.0, None

        results = yolo(image)
        detected_plate = None
        detected_confidence = 0.0

        found = False
        best_unvalidated = None
        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                logging.info(f"YOLO detection confidence: {conf}")

                if conf < 0.5:
                    continue

                # Draw bounding box
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Extract plate region and run OCR
                plate_roi = image[y1:y2, x1:x2]
                
                # Check if ROI is valid (has non-zero width and height)
                if plate_roi.size == 0 or plate_roi.shape[0] < 5 or plate_roi.shape[1] < 5:
                    logging.warning(f"Detected ROI too small: {plate_roi.shape}. Skipping OCR.")
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
                        cv2.putText(output_image, f"Plate: {detected_plate}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (255, 255, 0), 2)
                        found = True
                        break
                    else:
                        # See if a valid substring can be found inside the OCR candidate
                        inner = find_plate_in_text(candidate)
                        if inner and is_valid_plate_text(inner):
                            detected_plate = format_plate_text(inner)
                            detected_confidence = conf
                            cv2.putText(output_image, f"Plate: {detected_plate}",
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, (255, 255, 0), 2)
                            found = True
                            break
                        logging.warning(f"OCR returned text but failed validation: {candidate} (not a valid plate format)")
            if found:
                break
        # If we didn't find a validated plate, fall back to best unvalidated candidate
        if not detected_plate and best_unvalidated:
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

        # Save processed image
        processed_dir = os.path.join("static", "processed")
        os.makedirs(processed_dir, exist_ok=True)
        processed_image_path = os.path.join(processed_dir, f"processed_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(processed_image_path, output_image)
        logging.info(f"Processed image saved: {processed_image_path}")

        return detected_plate, detected_confidence, processed_image_path

    except Exception as e:
        logging.error(f"Detection failed: {str(e)}", exc_info=True)
        return None, 0.0, None
