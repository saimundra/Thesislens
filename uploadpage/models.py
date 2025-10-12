from django.db import models

class UploadedFile(models.Model):
    title = models.CharField(max_length=100)
    file =  models.FileField(upload_to="uploads")
    extracted_text = models.TextField(blank=True ,null=True)
    summary = models.TextField(blank=True ,null=True)
    grammar_issues = models.TextField(blank=True ,null=True)
    citations_issues = models.TextField(blank=True ,null=True)
    improvement_suggestions = models.TextField(blank=True ,null=True)
    uploaded_at =   models.DateTimeField(auto_now_add=True)
    

    def __str__(self):
        return self.title
