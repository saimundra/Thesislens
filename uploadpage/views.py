from django.shortcuts import render, redirect
from django.contrib import messages
from .models import UploadedFile
import os
import fitz         # PyMuPDF
from django.conf import settings
from docx import Document  # optional, only if you pip installed python-docx

def uploadpage(request):
    if request.method == "POST":
        title = request.POST.get("title", "mythesis").strip()
        uploaded_file = request.FILES.get("file")

        print(f"DEBUG: Title = {title}")
        print(f"DEBUG: File = {uploaded_file}")
        print(f"DEBUG: File name = {uploaded_file.name if uploaded_file else 'None'}")
        print(f"DEBUG: File size = {uploaded_file.size if uploaded_file else 'None'}")

        if not uploaded_file:
            messages.error(request, "No file selected!")
            return render(request, 'upload.html')

        # Optional: basic extension / content type check
        filename = uploaded_file.name.lower()
        if not (filename.endswith('.pdf') or filename.endswith('.docx') or filename.endswith('.doc')):
            messages.error(request, "Unsupported file type. Please upload PDF or DOC/DOCX.")
            return render(request, 'upload.html')

        try:
            # Save uploaded file to model & media folder
            uploaded_record = UploadedFile.objects.create(title=title, file=uploaded_file)
            print(f"DEBUG: File saved with ID: {uploaded_record.id}, path: {uploaded_record.file.path}")

            extracted_text = ""

            # If PDF -> use PyMuPDF
            if filename.endswith('.pdf'):
                pdf_path = uploaded_record.file.path
                # open pdf and extract text page by page
                with fitz.open(pdf_path) as doc:
                    for page in doc:
                        # get_text() returns text; for more control you can use "text" or "blocks"
                        extracted_text += page.get_text()

            # If DOCX -> use python-docx to extract text
            elif filename.endswith('.docx'):
                docx_path = uploaded_record.file.path
                doc = Document(docx_path)
                paragraphs = [p.text for p in doc.paragraphs]
                extracted_text = "\n".join(paragraphs)

            # If old .doc, you may need antiword or convert to docx server-side (optional)
            else:
                # fallback: try reading bytes (not guaranteed). You can add conversion here.
                extracted_text = ""

            # Save extracted text back to DB
            uploaded_record.extracted_text = extracted_text
            uploaded_record.save()
            print(f"DEBUG: Extracted text length = {len(extracted_text)}")

            messages.success(request, f"File '{uploaded_file.name}' uploaded and processed successfully!")

            # Render the same upload page and show a preview of extracted text
            return render(request, 'upload.html', {
                'success': True,
                'extracted_text': extracted_text[:4000],   # preview first 4k chars
                'uploaded_record': uploaded_record,
            })

        except Exception as e:
            print(f"DEBUG: Error saving/extracting file: {e}")
            messages.error(request, f"Error uploading or processing file: {e}")
            return render(request, 'upload.html')

    return render(request, 'upload.html')
