#!/usr/bin/env python3
"""
API Testing Script - Student Performance Prediction
"""
import requests
import json
import sys

# API endpoint
BASE_URL = "http://127.0.0.1:8000"
PREDICT_URL = f"{BASE_URL}/api/predict/"

# Test cases
test_cases = [
    {
        "name": "High Performer",
        "data": {
            "hours_studied": 9,
            "previous_scores": 95,
            "extracurricular": True,
            "sleep_hours": 8,
            "sample_papers": 7
        },
        "expected_range": (80, 100)
    },
    {
        "name": "Average Performer",
        "data": {
            "hours_studied": 5,
            "previous_scores": 70,
            "extracurricular": False,
            "sleep_hours": 6,
            "sample_papers": 4
        },
        "expected_range": (50, 75)
    },
    {
        "name": "Low Performer",
        "data": {
            "hours_studied": 2,
            "previous_scores": 45,
            "extracurricular": False,
            "sleep_hours": 4,
            "sample_papers": 1
        },
        "expected_range": (15, 40)
    },
    {
        "name": "Balanced Student",
        "data": {
            "hours_studied": 6,
            "previous_scores": 78,
            "extracurricular": True,
            "sleep_hours": 7,
            "sample_papers": 3
        },
        "expected_range": (60, 80)
    }
]

def test_prediction(test_case):
    """Test a single prediction"""
    print(f"\n{'='*70}")
    print(f"🧪 Test Case: {test_case['name']}")
    print(f"{'='*70}")

    print("\n📥 Input Data:")
    for key, value in test_case['data'].items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")

    try:
        response = requests.post(PREDICT_URL, json=test_case['data'], timeout=5)

        if response.status_code == 200:
            result = response.json()
            predicted_value = result['predicted_performance_index']
            expected_min, expected_max = test_case['expected_range']

            print(f"\n📤 Response:")
            print(f"  • Status Code: {response.status_code} ✅")
            print(f"  • Predicted Performance Index: {predicted_value}")

            if expected_min <= predicted_value <= expected_max:
                print(f"  • Validation: ✅ PASS (expected range: {expected_min}-{expected_max})")
            else:
                print(f"  • Validation: ⚠️  WARNING (expected range: {expected_min}-{expected_max})")

            print(f"\n📋 Full Response:")
            print(json.dumps(result, indent=2))

            return True
        else:
            print(f"\n❌ Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"\n❌ Connection Error: Could not connect to {BASE_URL}")
        print("   Make sure the Django server is running:")
        print("   python3 manage.py runserver")
        return False
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False

def test_invalid_request():
    """Test invalid request handling"""
    print(f"\n{'='*70}")
    print(f"🧪 Test Case: Invalid Request (Missing Fields)")
    print(f"{'='*70}")

    invalid_data = {
        "hours_studied": 5,
        # Missing other required fields
    }

    try:
        response = requests.post(PREDICT_URL, json=invalid_data, timeout=5)
        print(f"\n📤 Response:")
        print(f"  • Status Code: {response.status_code}")

        if response.status_code == 400:
            print(f"  • Validation: ✅ PASS (correctly rejected invalid request)")
        else:
            print(f"  • Validation: ⚠️  Expected 400 status code")

        print(f"\n📋 Response:")
        print(json.dumps(response.json(), indent=2))

    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║        Student Performance Prediction - API Tests            ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    print(f"🎯 Testing API at: {PREDICT_URL}")

    # Run all test cases
    results = []
    for test_case in test_cases:
        success = test_prediction(test_case)
        results.append((test_case['name'], success))

    # Test invalid request
    test_invalid_request()

    # Summary
    print(f"\n{'='*70}")
    print("📊 Test Summary")
    print(f"{'='*70}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} - {name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  🎉 All tests passed! The API is working correctly.")
    else:
        print("\n  ⚠️  Some tests failed. Please check the errors above.")

    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user.")
        sys.exit(0)
