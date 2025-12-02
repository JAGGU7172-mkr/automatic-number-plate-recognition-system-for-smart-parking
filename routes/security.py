from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from flask_login import login_required, current_user
from models import db
from forms import ScanVehicleForm
from models import Vehicle, VehicleLog
from anpr_utils import allowed_file, save_uploaded_file, detect_license_plate
from sqlalchemy import desc
import os
from datetime import datetime
import logging

security_bp = Blueprint('security', __name__, url_prefix='/security')

@security_bp.route('/scan', methods=['GET', 'POST'])
@login_required
def scan():
    form = ScanVehicleForm()
    sound_status = None
    auto_event_type = None  # used to send detected entry/exit to UI

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename, current_app.config['ALLOWED_EXTENSIONS']):
            try:
                saved_path = save_uploaded_file(file, current_app.config['UPLOAD_FOLDER'])
                if not saved_path:
                    flash("Failed to save uploaded file", "danger")
                    return redirect(request.url)

                # Process image
                plate_number, confidence, processed_relative_path = detect_license_plate(saved_path)

                if plate_number:
                    vehicle = Vehicle.query.filter_by(plate_number=plate_number).first()

                    # Determine auto entry/exit
                    # Determine auto entry/exit (Fixed Logic)
                    last_log = VehicleLog.query.filter_by(plate_number=plate_number).order_by(desc(VehicleLog.timestamp)).first()
                    if not last_log or last_log.event_type == "exit":
                        event_type = "entry"
                    elif last_log.event_type == "entry":
                        event_type = "exit"

                    auto_event_type = event_type

                    # Save vehicle log
                    log_entry = VehicleLog(
                        plate_number=plate_number,
                        vehicle_id=vehicle.id if vehicle else None,
                        is_authorized=vehicle.is_authorized if vehicle else False,
                        confidence=confidence,
                        event_type=event_type,
                        image_path=processed_relative_path,
                        processed_by=current_user.id,
                        timestamp=datetime.now()
                    )

                    db.session.add(log_entry)
                    db.session.commit()

                    if vehicle and vehicle.is_authorized:
                        flash(f"✅ Vehicle {plate_number} is authorized ({event_type}).", "success")
                        sound_status = "success"
                    else:
                        flash(f"🚫 Vehicle {plate_number} is not authorized ({event_type}).", "warning")
                        sound_status = "error"
                else:
                    flash("No license plate detected. Tips: make sure the plate is centered and fills at least ~20% of the image, keep the camera steady, avoid strong glare or reflections, and take the photo from a more direct angle. Try a closer crop and retry.", "warning")
                    sound_status = "error"

            except Exception as e:
                db.session.rollback()
                flash(f"Error processing image: {str(e)}", "danger")
                logging.exception("Exception during scanning")
                sound_status = "error"
        else:
            flash("Invalid file type.", "danger")

    recent_logs = VehicleLog.query.order_by(VehicleLog.timestamp.desc()).limit(10).all()
    return render_template('security/scan.html', form=form, recent_logs=recent_logs,
sound_status=sound_status, auto_event_type=auto_event_type)
