from django.db import models

# Create your models here.
class Contact(models.Model):
    first_name = models.CharField(max_length=255)
    last_name = models.CharField(max_length=255)
    email = models.EmailField(max_length=255)
    phone_number= models.CharField(max_length=10,unique=True)
    subject = models.CharField(max_length=100)
    description = models.CharField(max_length=500)