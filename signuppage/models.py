from django.db import models


# Create your models here.
class Signup(models.Model):
    full_name = models.CharField(max_length=255)
    email_address = models.EmailField(max_length=255)
    password = models.CharField(max_length=255)
    
    
