from rest_framework import serializers
from .models import StudentPerformance

class StudentPerformanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = StudentPerformance
        fields = [
            'id', 'student_name', 'hours_studied', 'previous_scores',
            'extracurricular', 'sleep_hours', 'sample_papers',
            'performance_index', 'predicted_index'
        ]
        read_only_fields = ['predicted_index']
