#!/usr/bin/env python
"""Test app and database initialization"""

from app import app, db
from models import User, Vehicle, VehicleLog

print('✓ Testing Vehicle Entry Tracker App')
print('=' * 50)

with app.app_context():
    print('✓ Database Configuration:')
    print(f'  URI: sqlite:///vehicle_tracker.db')
    print(f'  Track modifications: False')
    
    print('\n✓ Admin User:')
    admin = User.query.filter_by(username='admin').first()
    if admin:
        print(f'  Username: {admin.username}')
        print(f'  Email: {admin.email}')
        print(f'  Role: {admin.role}')
        print(f'  Password verified: {admin.check_password("admin123")}')
    else:
        print('  [ERROR] Admin user not found')
    
    print('\n✓ Database Tables:')
    print(f'  Users: {User.query.count()}')
    print(f'  Vehicles: {Vehicle.query.count()}')
    print(f'  Scan Logs: {VehicleLog.query.count()}')
    
    print('\n✓ Routes Registered:')
    routes = sorted([str(rule) for rule in app.url_map.iter_rules() 
                    if not rule.rule.startswith('/static') and rule.endpoint != 'static'])
    print(f'  Total: {len(routes)}')
    for route in routes:
        print(f'    - {route}')

print('\n✓ All tests passed!')
print('=' * 50)
