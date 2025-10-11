from django.db import models

class Contact(models.Model):
    
    first_name = models.CharField(max_length=255)
    last_name = models.CharField(max_length=255)
    email = models.EmailField(max_length=255)
    phone_number = models.CharField(max_length=10, unique=True)
    subject = models.CharField(max_length=100)
    description = models.CharField(max_length=500)

    def __str__(self):
        return f"{self.first_name} {self.last_name}"
