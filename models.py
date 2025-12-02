from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize SQLAlchemy
db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'admin' or 'security'
    created_at = db.Column(db.DateTime, default=lambda: datetime.now())  # LOCAL TIME

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def is_admin(self):
        return self.role == 'admin'
    
    def is_security(self):
        return self.role == 'security'
    
    def __repr__(self):
        return f'<User {self.username}>'

class Vehicle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plate_number = db.Column(db.String(20), unique=True, nullable=False)
    owner_name = db.Column(db.String(100), nullable=True)
    vehicle_type = db.Column(db.String(50), nullable=True)
    is_authorized = db.Column(db.Boolean, default=True)
    added_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now())  # LOCAL TIME
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(), onupdate=lambda: datetime.now())  # LOCAL TIME
    
    user = db.relationship('User', backref=db.backref('vehicles', lazy=True))
    
    def __repr__(self):
        return f'<Vehicle {self.plate_number}>'

class VehicleLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    vehicle_id = db.Column(db.Integer, db.ForeignKey('vehicle.id'), nullable=True)
    plate_number = db.Column(db.String(20), nullable=False)
    is_authorized = db.Column(db.Boolean, default=False)
    confidence = db.Column(db.Float, nullable=True)  # Confidence score of ANPR
    event_type = db.Column(db.String(10), nullable=False)  # 'entry' or 'exit'
    image_path = db.Column(db.String(255), nullable=True)
    processed_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now())  # LOCAL TIME
    
    vehicle = db.relationship('Vehicle', backref=db.backref('logs', lazy=True))
    user = db.relationship('User', backref=db.backref('processed_logs', lazy=True))
    
    def __repr__(self):
        return f'<VehicleLog {self.plate_number} {self.event_type} at {self.timestamp}>'
