from django.urls import path
from . import views 
urlpatterns =[
    path("",views.uploadpage,name= "uploadpage"),
]
