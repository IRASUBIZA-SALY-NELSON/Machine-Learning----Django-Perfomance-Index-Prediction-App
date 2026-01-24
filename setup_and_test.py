#!/usr/bin/env python3
"""
Test script for Student Performance Prediction API
"""
import subprocess
import sys
import time

def run_command(command, description):
    """Run a shell command and display results"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print()

    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.stdout:
        print("✅ OUTPUT:")
        print(result.stdout)

    if result.stderr:
        print("⚠️  STDERR:")
        print(result.stderr)

    return result.returncode == 0

def main():
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║   Student Performance Prediction - Setup & Test Script    ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    # Step 1: Check Python version
    print("\n📌 Step 1: Checking Python version...")
    run_command("python3 --version", "Python Version Check")

    # Step 2: Load data into database
    print("\n📌 Step 2: Loading data into database...")
    load_script = """
from performance.load_data import run
run()
"""
    with open('temp_load.py', 'w') as f:
        f.write(load_script)

    run_command(
        "python3 manage.py shell < temp_load.py",
        "Loading CSV data into database"
    )

    # Step 3: Train the model
    print("\n📌 Step 3: Training the machine learning model...")
    train_script = """
from performance.train_model import train
train()
"""
    with open('temp_train.py', 'w') as f:
        f.write(train_script)

    run_command(
        "python3 manage.py shell < temp_train.py",
        "Training Random Forest model"
    )

    # Clean up temp files
    subprocess.run("rm -f temp_load.py temp_train.py", shell=True)

    print("""

    ╔════════════════════════════════════════════════════════════╗
    ║                    Setup Complete! ✅                       ║
    ╚════════════════════════════════════════════════════════════╝

    📝 Next Steps:

    1. Start the development server:
       python3 manage.py runserver

    2. Test the API using curl:
       curl -X POST http://127.0.0.1:8000/api/predict/ \\
         -H "Content-Type: application/json" \\
         -d '{
           "hours_studied": 6,
           "previous_scores": 78,
           "extracurricular": true,
           "sleep_hours": 7,
           "sample_papers": 3
         }'

    3. Or use the test_api.py script:
       python3 test_api.py

    📚 For more information, see README.md

    """)

if __name__ == "__main__":
    main()
