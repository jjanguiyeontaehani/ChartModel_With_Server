from django.contrib import admin
from django.urls import path

from view import views as index_views
import repositories

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index_views.index),
]
