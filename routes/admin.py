from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
from models import db
from forms import UserForm, VehicleForm
from models import User, Vehicle
from functools import wraps
import logging

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# Custom decorator to restrict access to admin users
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin():
            flash('Access denied. Admin privileges required.', 'danger')
            return redirect(url_for('dashboard.index'))
        return f(*args, **kwargs)
    return decorated_function

@admin_bp.route('/')
@login_required
@admin_required
def index():
    return redirect(url_for('admin.vehicles'))

@admin_bp.route('/vehicles', methods=['GET'])
@login_required
@admin_required
def vehicles():
    # Get filter parameter from query string
    filter_param = request.args.get('filter', '')
    
    # Apply filters if specified
    if filter_param == 'authorized':
        vehicles = Vehicle.query.filter_by(is_authorized=True).order_by(Vehicle.created_at.desc()).all()
        filter_title = "Authorized Vehicles"
    elif filter_param == 'unauthorized':
        vehicles = Vehicle.query.filter_by(is_authorized=False).order_by(Vehicle.created_at.desc()).all()
        filter_title = "Unauthorized Vehicles"
    else:
        vehicles = Vehicle.query.order_by(Vehicle.created_at.desc()).all()
        filter_title = "All Vehicles"
    
    form = VehicleForm()
    return render_template('admin/vehicles.html', vehicles=vehicles, form=form, 
                          filter_title=filter_title, current_filter=filter_param)

@admin_bp.route('/vehicles/add', methods=['POST'])
@login_required
@admin_required
def add_vehicle():
    form = VehicleForm()
    if form.validate_on_submit():
        try:
            vehicle = Vehicle(
                plate_number=form.plate_number.data.upper(),
                owner_name=form.owner_name.data,
                vehicle_type=form.vehicle_type.data,
                is_authorized=form.is_authorized.data,
                added_by=current_user.id
            )
            db.session.add(vehicle)
            db.session.commit()
            flash(f'Vehicle with plate {vehicle.plate_number} has been added!', 'success')
            logging.info(f"Vehicle {vehicle.plate_number} added by {current_user.username}")
            return redirect(url_for('admin.vehicles'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error adding vehicle: {str(e)}', 'danger')
            logging.error(f"Error adding vehicle: {str(e)}")
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f'{getattr(form, field).label.text}: {error}', 'danger')
    
    return redirect(url_for('admin.vehicles'))

@admin_bp.route('/vehicles/<int:vehicle_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_vehicle(vehicle_id):
    vehicle = Vehicle.query.get_or_404(vehicle_id)
    try:
        db.session.delete(vehicle)
        db.session.commit()
        flash(f'Vehicle with plate {vehicle.plate_number} has been deleted!', 'success')
        logging.info(f"Vehicle {vehicle.plate_number} deleted by {current_user.username}")
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting vehicle: {str(e)}', 'danger')
        logging.error(f"Error deleting vehicle: {str(e)}")
    
    return redirect(url_for('admin.vehicles'))

@admin_bp.route('/vehicles/<int:vehicle_id>/toggle', methods=['POST'])
@login_required
@admin_required
def toggle_vehicle_authorization(vehicle_id):
    vehicle = Vehicle.query.get_or_404(vehicle_id)
    try:
        vehicle.is_authorized = not vehicle.is_authorized
        db.session.commit()
        status = "authorized" if vehicle.is_authorized else "unauthorized"
        flash(f'Vehicle with plate {vehicle.plate_number} is now {status}!', 'success')
        logging.info(f"Vehicle {vehicle.plate_number} status changed to {status} by {current_user.username}")
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating vehicle: {str(e)}', 'danger')
        logging.error(f"Error updating vehicle: {str(e)}")
    
    return redirect(url_for('admin.vehicles'))

@admin_bp.route('/users', methods=['GET'])
@login_required
@admin_required
def users():
    users = User.query.order_by(User.created_at.desc()).all()
    form = UserForm()
    return render_template('admin/users.html', users=users, form=form)

@admin_bp.route('/users/add', methods=['POST'])
@login_required
@admin_required
def add_user():
    form = UserForm()
    if form.validate_on_submit():
        try:
            user = User(
                username=form.username.data,
                email=form.email.data,
                role=form.role.data
            )
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            flash(f'User {user.username} has been added!', 'success')
            logging.info(f"User {user.username} added by {current_user.username}")
            return redirect(url_for('admin.users'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error adding user: {str(e)}', 'danger')
            logging.error(f"Error adding user: {str(e)}")
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f'{getattr(form, field).label.text}: {error}', 'danger')
    
    return redirect(url_for('admin.users'))

@admin_bp.route('/users/<int:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    # Prevent self-deletion
    if user_id == current_user.id:
        flash('You cannot delete your own account!', 'danger')
        return redirect(url_for('admin.users'))
    
    user = User.query.get_or_404(user_id)
    try:
        db.session.delete(user)
        db.session.commit()
        flash(f'User {user.username} has been deleted!', 'success')
        logging.info(f"User {user.username} deleted by {current_user.username}")
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting user: {str(e)}', 'danger')
        logging.error(f"Error deleting user: {str(e)}")
    
    return redirect(url_for('admin.users'))
