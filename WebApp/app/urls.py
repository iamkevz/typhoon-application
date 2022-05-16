from django.urls import path
from . import views

urlpatterns = [
    path('',views.req, name='app-index'),
    path("render_map",views.render_map),
]