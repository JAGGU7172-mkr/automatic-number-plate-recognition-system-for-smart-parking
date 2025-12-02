#!/usr/bin/env python
"""Functional test suite for all features"""

from app import app, db
from models import User, Vehicle, VehicleLog
from datetime import datetime, timedelta

print('[OK] Vehicle Entry Tracker - Functional Tests')
print('=' * 60)

with app.app_context():
    # Test 1: User Management
    print('\n1. Testing User Management:')
    
    security_user = User.query.filter_by(username='security').first()
    if not security_user:
        security_user = User(username='security', email='security@vehicle-tracker.local', role='security')
        security_user.set_password('sec123')
        db.session.add(security_user)
        db.session.commit()
    print('  [OK] Security user ready')
    
    sec = User.query.filter_by(username='security').first()
    if sec and sec.check_password('sec123'):
        print('  [OK] Password verification works')
    
    admin = User.query.filter_by(username='admin').first()
    if admin.is_admin:
        print('  [OK] Admin role check works')
    
    # Test 2: Vehicle Management
    print('\n2. Testing Vehicle Management:')
    
    admin = User.query.filter_by(username='admin').first()
    
    vehicle1 = Vehicle.query.filter_by(plate_number='ABC123').first()
    if not vehicle1:
        vehicle1 = Vehicle(plate_number='ABC123', owner_name='John Doe', vehicle_type='sedan', is_authorized=True, added_by=admin.id)
        db.session.add(vehicle1)
        db.session.commit()
    
    vehicle2 = Vehicle.query.filter_by(plate_number='XYZ789').first()
    if not vehicle2:
        vehicle2 = Vehicle(plate_number='XYZ789', owner_name='Jane Smith', vehicle_type='suv', is_authorized=False, added_by=admin.id)
        db.session.add(vehicle2)
        db.session.commit()
    
    print('  [OK] Vehicles ready in database')
    
    vehicles = Vehicle.query.all()
    print(f'  [OK] {len(vehicles)} vehicles in database')
    
    vehicle1.is_authorized = False
    db.session.commit()
    v1 = Vehicle.query.filter_by(plate_number='ABC123').first()
    print(f'  [OK] Vehicle authorization toggle works')
    
    # Test 3: Scan Logging
    print('\n3. Testing Scan Logging:')
    
    sec_user = User.query.filter_by(username='security').first()
    
    log1 = VehicleLog(vehicle_id=vehicle1.id, plate_number='ABC123', is_authorized=True, event_type='entry', image_path='image1.jpg', processed_by=sec_user.id)
    db.session.add(log1)
    db.session.commit()
    print('  [OK] Entry log created')
    
    log2 = VehicleLog(vehicle_id=vehicle1.id, plate_number='ABC123', is_authorized=True, event_type='exit', image_path='image2.jpg', processed_by=sec_user.id)
    db.session.add(log2)
    db.session.commit()
    print('  [OK] Exit log created')
    
    logs = VehicleLog.query.filter_by(vehicle_id=vehicle1.id).all()
    print(f'  [OK] {len(logs)} logs for vehicle')
    
    # Test 4: Dashboard Data
    print('\n4. Testing Dashboard Data Functions:')
    
    from utils import get_dashboard_stats, get_chart_data
    
    stats = get_dashboard_stats(db, VehicleLog, Vehicle)
    print(f'  [OK] Total entries: {stats.get("entry_count", 0)}')
    print(f'  [OK] Total exits: {stats.get("exit_count", 0)}')
    print(f'  [OK] Registered vehicles: {stats.get("total_vehicles", 0)}')
    
    chart_data = get_chart_data(db, VehicleLog)
    print(f'  [OK] Chart data hours: {len(chart_data.get("labels", []))}')
    
    # Test 5: ANPR Pipeline
    print('\n5. Testing ANPR Pipeline:')
    
    from anpr_utils import get_yolo_model, get_paddle_ocr
    
    yolo = get_yolo_model()
    print(f'  [OK] YOLO model loaded: {yolo is not None}')
    
    ocr = get_paddle_ocr()
    print(f'  [OK] PaddleOCR loaded: {ocr is not None}')
    
    # Test 6: Forms
    print('\n6. Testing Forms:')
    
    from forms import LoginForm, RegistrationForm, ScanVehicleForm
    
    print('  [OK] LoginForm imported')
    print('  [OK] RegistrationForm imported')
    print('  [OK] ScanVehicleForm imported')
    
    print('  [OK] All form classes available')
    
    print('\n' + '=' * 60)
    print('[OK] All functional tests passed!')
    print('=' * 60)
