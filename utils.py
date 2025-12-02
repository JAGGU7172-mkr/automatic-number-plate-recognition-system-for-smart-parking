import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import current_app
from sqlalchemy import func

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def save_image(file):
    """Save the uploaded image and return the path."""
    if file and allowed_file(file.filename):
        # Create a secure filename with a unique identifier
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        
        # Create directory if it doesn't exist
        today = datetime.now().strftime('%Y-%m-%d')  # LOCAL TIME
        upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], today)
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the file
        relative_path = os.path.join('uploads', today, unique_filename)
        file.save(os.path.join(current_app.root_path, 'static', relative_path))
        
        return relative_path
    return None

def format_datetime(dt):  
    """Format datetime using the computer's local time."""  
    if dt:
        if dt.tzinfo is None:  # Ensure dt is timezone-aware
            dt = dt.astimezone()
        return dt.strftime('%Y-%m-%d %H:%M:%S')  
    return "-"

def get_dashboard_stats(db, VehicleLog, Vehicle):
    """Get statistics for the dashboard."""
    total_vehicles = Vehicle.query.count()
    total_logs = VehicleLog.query.count()
    
    # Get count of logs for today in local time
    today = datetime.now().date()
    today_logs = VehicleLog.query.filter(
        db.func.date(VehicleLog.timestamp) == today
    ).count()
    
    # Get count of approved/blocked vehicles
    approved_vehicles = Vehicle.query.filter_by(is_authorized=True).count()
    blocked_vehicles = Vehicle.query.filter_by(is_authorized=False).count()
    
    # Get entry/exit counts for today
    entry_count = VehicleLog.query.filter(
        db.func.date(VehicleLog.timestamp) == today,
        VehicleLog.event_type == 'entry'
    ).count()
    
    exit_count = VehicleLog.query.filter(
        db.func.date(VehicleLog.timestamp) == today,
        VehicleLog.event_type == 'exit'
    ).count()
    
    # Get recognized vs. unrecognized counts
    recognized_count = VehicleLog.query.filter_by(is_authorized=True).count()
    unrecognized_count = VehicleLog.query.filter_by(is_authorized=False).count()
    
    # Get recent logs (last 10)
    recent_logs = VehicleLog.query.order_by(VehicleLog.timestamp.desc()).limit(10).all()
    
    return {
        'total_vehicles': total_vehicles,
        'total_logs': total_logs,
        'today_logs': today_logs,
        'approved_vehicles': approved_vehicles,
        'blocked_vehicles': blocked_vehicles,
        'entry_count': entry_count,
        'exit_count': exit_count,
        'recognized_count': recognized_count,
        'unrecognized_count': unrecognized_count,
        'recent_logs': recent_logs
    }



def get_chart_data(db, VehicleLog):
    """Return hourly entry/exit counts for grouped bar chart."""
    today = datetime.now().date()

    # Query all logs for today
    today_logs = VehicleLog.query.filter(
        func.date(VehicleLog.timestamp) == today
    ).order_by(VehicleLog.timestamp.asc()).all()

    # Initialize 24-hour slots
    entry_counts = [0] * 24
    exit_counts = [0] * 24

    for log in today_logs:
        hour = log.timestamp.hour
        if log.event_type == 'entry' and log.is_authorized == True:
            entry_counts[hour] += 1
        elif log.event_type == 'exit' and log.is_authorized == True:
            exit_counts[hour] += 1

    hours = [f"{h:02d}:00" for h in range(24)]

    return {
        'labels': hours,
        'entries': entry_counts,
        'exits': exit_counts
    }

