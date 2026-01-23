#!/usr/bin/env python3
"""
Quick start script for the Student Performance Predictor
Run this file to start the web application
"""

import os
import sys

def main():
    print("=" * 60)
    print("🎓 STUDENT PERFORMANCE PREDICTOR - NAIVE BAYES")
    print("=" * 60)
    print()
    
    # Check if required packages are installed
    try:
        import flask
        import pandas
        import sklearn
        print("✅ All required packages are installed")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return
    
    print("🚀 Starting the web application...")
    print("📊 Training the Naive Bayes model...")
    
    # Import and run the Flask app
    from app import app
    
    print()
    print("🌐 Web application is running!")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    print()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()