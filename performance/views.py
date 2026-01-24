from rest_framework import viewsets
from .models import StudentPerformance
from .serializers import StudentPerformanceSerializer
from django.shortcuts import render

class StudentPerformanceViewSet(viewsets.ModelViewSet):
    queryset = StudentPerformance.objects.all().order_by('-id')
    serializer_class = StudentPerformanceSerializer

def teacher_portal(request):
    """View to serve the Teacher Portal UI"""
    return render(request, 'performance/portal.html')
