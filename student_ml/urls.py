from django.contrib import admin
from django.urls import path, include
from performance.views import teacher_portal

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', teacher_portal, name='portal'),
    path('api/', include('performance.urls')),
]
