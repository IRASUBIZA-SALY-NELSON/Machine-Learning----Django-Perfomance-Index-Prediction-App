import joblib
import numpy as np
import pandas as pd
import os
from django.db import models

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

class StudentPerformance(models.Model):
    student_name = models.CharField(max_length=100, default="Unknown Student")
    hours_studied = models.IntegerField()
    previous_scores = models.IntegerField()
    extracurricular = models.BooleanField()
    sleep_hours = models.IntegerField()
    sample_papers = models.IntegerField()
    performance_index = models.FloatField(default=0.0) # Actual score
    predicted_index = models.FloatField(null=True, blank=True)

    def save(self, *args, **kwargs):
        # Auto-predict using the model if it exists
        if os.path.exists(MODEL_PATH):
            try:
                model = joblib.load(MODEL_PATH)
                features = pd.DataFrame([{
                    'Hours Studied': self.hours_studied,
                    'Previous Scores': self.previous_scores,
                    'Extracurricular Activities': 1 if self.extracurricular else 0,
                    'Sleep Hours': self.sleep_hours,
                    'Sample Question Papers Practiced': self.sample_papers,
                }])
                self.predicted_index = round(float(model.predict(features)[0]), 2)
            except Exception as e:
                print(f"Error predicting: {e}")

        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.student_name} - Pred: {self.predicted_index}"
