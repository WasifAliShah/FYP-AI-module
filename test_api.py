#!/usr/bin/env python3
"""
Test script for the Face Verification FastAPI application
"""
import requests
import json
import time
import os

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_root():
    """Test the root endpoint"""
    print("\nTesting root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_image_verification():
    """Test image verification endpoint"""
    print("\nTesting image verification...")
    
    # Check if test image exists
    if not os.path.exists("wasif7.jpg"):
        print("Test image 'wasif7.jpg' not found. Skipping image verification test.")
        return False
    
    try:
        with open("wasif7.jpg", "rb") as f:
            files = {"file": ("wasif7.jpg", f, "image/jpeg")}
            data = {"threshold": 0.5, "cooldown": 5}
            
            response = requests.post(f"{BASE_URL}/verify-image", files=files, data=data)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_video_processing():
    """Test video processing endpoint"""
    print("\nTesting video processing...")
    
    # Check if test video exists
    if not os.path.exists("combine.mp4"):
        print("Test video 'combine.mp4' not found. Skipping video processing test.")
        return False
    
    try:
        data = {
            "video_path": "combine.mp4",
            "reference_image_path": "wasif7.jpg",
            "threshold": 0.5,
            "cooldown": 5,
            "skip_frames": 10
        }
        
        response = requests.post(f"{BASE_URL}/process-video", json=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("Face Verification API Test Suite")
    print("=" * 40)
    
    tests = [
        ("Health Check", test_health),
        ("Root Endpoint", test_root),
        ("Image Verification", test_image_verification),
        ("Video Processing", test_video_processing)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
        time.sleep(1)  # Small delay between tests
    
    print(f"\n{'='*20} Test Results {'='*20}")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")

if __name__ == "__main__":
    main()
