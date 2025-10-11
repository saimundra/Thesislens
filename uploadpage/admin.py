from django.contrib import admin
from .models import UploadedFile

# Register your models here.
@admin.register(UploadedFile)
class UploadedFileAdmin(admin.ModelAdmin):
    list_display = ['title', 'file', 'uploaded_at']
    list_filter = ['uploaded_at']
    search_fields = ['title']
