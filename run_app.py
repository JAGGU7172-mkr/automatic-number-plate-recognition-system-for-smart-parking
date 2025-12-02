#!/usr/bin/env python
"""
Startup script with optimized settings to avoid excessive reloading
"""
from app import app

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
