import pandas as pd
from performance.models import StudentPerformance

def run():
    """Load data from CSV file into the database"""
    df = pd.read_csv('StudentPerformance.csv')

    # Clear existing data
    StudentPerformance.objects.all().delete()

    for _, row in df.iterrows():
        StudentPerformance.objects.create(
            hours_studied=row['Hours Studied'],
            previous_scores=row['Previous Scores'],
            extracurricular=row['Extracurricular Activities'] == 'Yes',
            sleep_hours=row['Sleep Hours'],
            sample_papers=row['Sample Question Papers Practiced'],
            performance_index=row['Performance Index'],
        )

    print(f"Successfully loaded {len(df)} records into the database")
