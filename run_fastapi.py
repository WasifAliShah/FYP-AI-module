#!/usr/bin/env python3
"""
Script to run the FastAPI application with proper virtual environment activation
"""
import subprocess
import sys
import os

def run_fastapi():
    """Run the FastAPI application"""
    print("Starting Face Verification FastAPI Application...")
    print("=" * 50)
    
    # Check if we're in the virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Warning: Virtual environment not detected")
        print("Please activate your virtual environment first:")
        print("  Windows: venv\\Scripts\\activate")
        print("  Linux/Mac: source venv/bin/activate")
        print()
    
    # Check if required files exist
    required_files = ["main.py", "wasif7.jpg", "yolov8n-face-lindevs.pt"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found")
    print()
    
    try:
        # Run the FastAPI application
        print("🚀 Starting FastAPI server...")
        print("📡 API will be available at: http://localhost:8000")
        print("📚 API documentation at: http://localhost:8000/docs")
        print("🔄 Press Ctrl+C to stop the server")
        print()
        
        # Use uvicorn to run the FastAPI app
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    run_fastapi()
