from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField, PasswordField, SubmitField, SelectField, BooleanField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError
from models import User, Vehicle

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Log In')
    
class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm Password', 
                                    validators=[DataRequired(), EqualTo('password')])
    role = SelectField('User Role', choices=[('admin', 'Admin'), ('security', 'Security')],
                       validators=[DataRequired()], 
                       description='Admin can add and manage license plates, Security can only scan vehicles')
    submit = SubmitField('Register')
    
    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Username already taken. Please choose a different one.')
    
    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email already registered. Please use a different one.')

class UserForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm Password', 
                                     validators=[DataRequired(), EqualTo('password')])
    role = SelectField('Role', choices=[('admin', 'Admin'), ('security', 'Security')],
                       validators=[DataRequired()])
    submit = SubmitField('Add User')
    
    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Username already taken. Please choose a different one.')
    
    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email already registered. Please use a different one.')

class VehicleForm(FlaskForm):
    plate_number = StringField('License Plate Number', validators=[DataRequired()])
    owner_name = StringField('Owner Name', validators=[DataRequired()])
    vehicle_type = SelectField('Vehicle Type', 
                              choices=[('car', 'Car'), ('truck', 'Truck'), 
                                       ('motorcycle', 'Motorcycle'), ('other', 'Other')],
                              validators=[DataRequired()])
    is_authorized = BooleanField('Authorized', default=True)
    submit = SubmitField('Add Vehicle')
    
    def validate_plate_number(self, plate_number):
        vehicle = Vehicle.query.filter_by(plate_number=plate_number.data).first()
        if vehicle:
            raise ValidationError('License plate already registered.')

class ScanVehicleForm(FlaskForm):
    image = FileField('Upload Image', 
                     validators=[FileRequired(), 
                                FileAllowed(['jpg', 'jpeg', 'png'], 'Images only!')])
    submit = SubmitField('Scan Vehicle')

class RequestPasswordResetForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Request Password Reset')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is None:
            raise ValidationError('There is no account with that email. You must register first.')

class ResetPasswordForm(FlaskForm):
    password = PasswordField('New Password', validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm New Password',
                                   validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Reset Password')
