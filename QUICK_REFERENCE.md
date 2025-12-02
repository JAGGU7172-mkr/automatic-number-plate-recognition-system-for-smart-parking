# Vehicle Entry Tracker - Quick Reference

## Access the Application
- **URL**: http://127.0.0.1:5000
- **Status**: ✓ Running
- **Port**: 5000

## Default Login
```
Username: admin
Password: admin123
```

## Main Features

### 1. Dashboard (/)
- Real-time statistics (vehicles, entries, exits, unauthorized)
- Hourly entry/exit chart
- Recent scan logs
- Top vehicles

### 2. Security Scan (/security/scan)
- Upload vehicle images
- Automatic license plate detection
- Auto entry/exit toggle
- Sound feedback
- Recent scans list

### 3. Admin Panel (/admin/)

#### Vehicles Management
- Add new vehicles (license plate + owner info)
- Toggle authorization status
- Delete vehicles
- View all registered vehicles

#### Users Management
- Add new security users
- Delete users
- View all users
- Assign roles (admin/security)

### 4. Logs (/vehicle-logs)
- View all scan history
- Filter by date
- Export functionality

## Key Files
- `app.py` - Main Flask app
- `models.py` - Database models
- `routes/` - API endpoints
- `templates/` - HTML pages
- `static/models/licence_plate.pt` - YOLO model
- `instance/vehicle_tracker.db` - SQLite database

## Environment
- Python 3.13
- Flask 2.x
- SQLite database
- YOLO + PaddleOCR ML models
- Bootstrap 5 UI

## How to Stop the App
Press `Ctrl+C` in the terminal where it's running.

## Startup Commands

### Production (Recommended)
```bash
python run_app.py
```

### Development
```bash
python main.py
```

## Test Suite
```bash
# App initialization tests
python test_app.py

# Comprehensive functional tests
python test_functional.py
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 5000 in use | Change port in `run_app.py` |
| Database errors | Delete `instance/vehicle_tracker.db` and restart |
| YOLO model missing | Ensure `static/models/licence_plate.pt` exists |
| PaddleOCR hangs | First run downloads models (~2-3 min) |

## Database Reset
```bash
Remove-Item instance/vehicle_tracker.db -Force
python run_app.py  # Recreates fresh database with default admin
```

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Dashboard |
| GET/POST | `/login` | Login |
| GET/POST | `/register` | Register |
| GET | `/logout` | Logout |
| GET/POST | `/security/scan` | Scan vehicle |
| GET | `/vehicle-logs` | View logs |
| GET | `/api/stats` | Get stats JSON |
| GET/POST | `/admin/vehicles` | Manage vehicles |
| POST | `/admin/vehicles/<id>/toggle` | Toggle authorization |
| POST | `/admin/vehicles/<id>/delete` | Delete vehicle |
| GET/POST | `/admin/users` | Manage users |
| POST | `/admin/users/<id>/delete` | Delete user |

## Status Summary
✓ Database: SQLite initialized
✓ Backend: All 15 routes operational
✓ ML Models: YOLO + PaddleOCR loaded
✓ Frontend: Bootstrap UI responsive
✓ Tests: All functional tests passing (100%)
✓ Server: Running on http://127.0.0.1:5000

Ready for use!
