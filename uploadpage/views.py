from django.shortcuts import render, redirect
from django.contrib import messages
from .models import UploadedFile
import os
import fitz  # PyMuPDF
from docx import Document
from services.gemini_client import analyze_thesis_text

def uploadpage(request):
    if request.method == "POST":
        title = request.POST.get("title", "mythesis").strip()
        uploaded_file = request.FILES.get("file")

        if not uploaded_file:
            messages.error(request, "No file selected!")
            return render(request, 'upload.html')

        filename = uploaded_file.name.lower()
        if not (filename.endswith('.pdf') or filename.endswith('.docx')):
            messages.error(request, "Unsupported file type. Please upload PDF or DOCX.")
            return render(request, 'upload.html')

        try:
            # Save file to DB
            uploaded_record = UploadedFile.objects.create(title=title, file=uploaded_file)
            extracted_text = ""

            # Extract text depending on file type
            if filename.endswith('.pdf'):
                with fitz.open(uploaded_record.file.path) as doc:
                    for page in doc:
                        extracted_text += page.get_text()
            elif filename.endswith('.docx'):
                doc = Document(uploaded_record.file.path)
                paragraphs = [p.text for p in doc.paragraphs]
                extracted_text = "\n".join(paragraphs)

            # Save extracted text
            uploaded_record.extracted_text = extracted_text
            uploaded_record.save()

            print("DEBUG: Extracted text length =", len(extracted_text))

            # 🔥 Send extracted text to Gemini for analysis
            try:
                analysis = analyze_thesis_text(extracted_text)
                print("DEBUG: Analysis returned:", analysis)
            except Exception as e:
                analysis = None
                print("DEBUG: Gemini analysis failed:", e)

            # Save analysis in DB
            if analysis:
                uploaded_record.summary = analysis.get('summary', '')
                uploaded_record.grammar_issues = analysis.get('grammar', '')
                uploaded_record.citations_issues = analysis.get('citations', '')
                uploaded_record.improvement_suggestions = analysis.get('improvements', '')
                uploaded_record.save()

            messages.success(request, f"File '{uploaded_file.name}' uploaded successfully!")

            # Render with analysis results
            return render(request, 'upload.html', {
                'success': True,
                'extracted_text': extracted_text[:4000],
                'analysis': analysis,
                'uploaded_record': uploaded_record,
            })

        except Exception as e:
            messages.error(request, f"Error uploading or analyzing file: {e}")
            print("DEBUG:", e)

    return render(request, 'upload.html')
