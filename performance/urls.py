from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import StudentPerformanceViewSet

router = DefaultRouter()
router.register(r'students', StudentPerformanceViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
