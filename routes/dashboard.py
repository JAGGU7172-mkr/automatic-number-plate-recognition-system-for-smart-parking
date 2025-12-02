from flask import Blueprint, render_template, jsonify
from flask_login import login_required, current_user
from models import db, Vehicle, VehicleLog
from sqlalchemy import func, desc
from datetime import datetime, timedelta
from utils import get_chart_data
import logging
import json

# Force logging to console and file
logger = logging.getLogger('dashboard')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
file_handler = logging.FileHandler('debug.log', mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.handlers = [handler, file_handler]

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
@login_required
def index():
    logger.info("Entering dashboard index route")
    
    try:
        # Vehicle statistics
        total_vehicles = Vehicle.query.count()
        authorized_vehicles = Vehicle.query.filter_by(is_authorized=True).count()
        unauthorized_vehicles = total_vehicles - authorized_vehicles
        
        # Today's logs
        today = datetime.now().date()  # Change from datetime.utcnow() to datetime.now()
        today_start = datetime.combine(today, datetime.min.time())
        today_logs = VehicleLog.query.filter(VehicleLog.timestamp >= today_start).count()
        today_entries = VehicleLog.query.filter(
            VehicleLog.timestamp >= today_start,
            VehicleLog.event_type == 'entry',
            VehicleLog.is_authorized == True
        ).count()
        today_exits = VehicleLog.query.filter(
            VehicleLog.timestamp >= today_start,
            VehicleLog.event_type == 'exit',
            VehicleLog.is_authorized == True
        ).count()
        
        # Unauthorized attempts
        today_unauthorized = VehicleLog.query.filter(
            VehicleLog.timestamp >= today_start,
            VehicleLog.is_authorized == False
        ).count()
        
        recent_logs = VehicleLog.query.order_by(VehicleLog.timestamp.desc()).limit(10).all()
        
        # Hourly chart data
        chart_data = get_chart_data(db, VehicleLog)
        date_labels = chart_data['labels']
        entry_data = chart_data['entries']
        exit_data = chart_data['exits']
        
        # Add current hour for debugging
        current_hour = datetime.now().hour
        logger.info(f"Current hour: {current_hour}")
        logger.info(f"Raw Chart Data - Labels: {date_labels}")
        logger.info(f"Raw Chart Data - Entries: {entry_data}")
        logger.info(f"Raw Chart Data - Exits: {exit_data}")
        
        # Check if there's data for the current hour
        if 0 <= current_hour < 24:
            logger.info(f"Entry data for hour {current_hour}: {entry_data[current_hour]}")
            logger.info(f"Exit data for hour {current_hour}: {exit_data[current_hour]}")
        
        # Serialize to JSON
        date_labels_json = json.dumps(date_labels)
        entry_data_json = json.dumps(entry_data)
        exit_data_json = json.dumps(exit_data)
        
        # Validate JSON strings
        try:
            # Test parse to ensure valid JSON
            json.loads(date_labels_json)
            json.loads(entry_data_json)
            json.loads(exit_data_json)
            logger.info("JSON validation successful")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON generated: {str(e)}")
            # Fallback to safe values
            date_labels_json = json.dumps([f"{i:02d}:00" for i in range(24)])
            entry_data_json = json.dumps([0] * 24)
            exit_data_json = json.dumps([0] * 24)
        
        logger.info(f"JSON Chart Data - Labels: {date_labels_json}")
        logger.info(f"JSON Chart Data - Entries: {entry_data_json}")
        logger.info(f"JSON Chart Data - Exits: {exit_data_json}")
        
        # Add debug flag
        debug_mode = True
        
        # Top 5 vehicles
        top_vehicles = db.session.query(
            VehicleLog.plate_number,
            func.count(VehicleLog.id).label('count')
        ).group_by(VehicleLog.plate_number).order_by(desc('count')).limit(5).all()
        
        logger.info("Rendering dashboard template")
        return render_template(
            'dashboard.html',
            total_vehicles=total_vehicles,
            authorized_vehicles=authorized_vehicles,
            unauthorized_vehicles=unauthorized_vehicles,
            today_logs=today_logs,
            today_entries=today_entries,
            today_exits=today_exits,
            today_unauthorized=today_unauthorized,  # Ensure this is included
            recent_logs=recent_logs,
            date_labels=date_labels_json,
            entry_data=entry_data_json,
            exit_data=exit_data_json,
            top_vehicles=top_vehicles,
            now=datetime.utcnow(),
            debug_mode=debug_mode
        )
    except Exception as e:
        logger.error(f"Error in dashboard index: {str(e)}", exc_info=True)
        # Fallback data
        return render_template(
            'dashboard.html',
            total_vehicles=0,
            authorized_vehicles=0,
            unauthorized_vehicles=0,
            today_logs=0,
            today_entries=0,
            today_exits=0,
            today_unauthorized=0,  # Ensure this is included
            recent_logs=[],
            date_labels=json.dumps([f"{i:02d}" for i in range(24)]),
            entry_data=json.dumps([0] * 24),
            exit_data=json.dumps([0] * 24),
            top_vehicles=[],
            now=datetime.utcnow()
        )

@dashboard_bp.route('/vehicle-logs')
@login_required
def vehicle_logs():
    logs = VehicleLog.query.order_by(VehicleLog.timestamp.desc()).all()
    return render_template('vehicle_logs.html', logs=logs)

@dashboard_bp.route('/api/stats')
@login_required
def get_stats():
    total_vehicles = Vehicle.query.count()
    authorized_vehicles = Vehicle.query.filter_by(is_authorized=True).count()
    today = datetime.utcnow().date()
    today_start = datetime.combine(today, datetime.min.time())
    today_logs = VehicleLog.query.filter(VehicleLog.timestamp >= today_start).count()
    return jsonify({
        'total_vehicles': total_vehicles,
        'authorized_vehicles': authorized_vehicles,
        'unauthorized_vehicles': total_vehicles - authorized_vehicles,
        'today_logs': today_logs
    })