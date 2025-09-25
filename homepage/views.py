from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from .models import Contact   


# Homepage view
def homepage(request):
    template = loader.get_template("homepage.html")
    return HttpResponse(template.render({}, request))


# Contact Us view
def contactus(request):
    template = loader.get_template("contactus.html")
    context = {}

    if request.method == 'POST':
        first_name = request.POST.get('firstName', '').strip()
        last_name = request.POST.get('lastName', '').strip()
        email = request.POST.get('email', '').strip()
        phone_number = request.POST.get('phone', '').strip()
        subject = request.POST.get('subject', '').strip()
        description = request.POST.get('description', '').strip()

        # ✅ Save to database
        Contact.objects.create(
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone_number=phone_number,
            subject=subject,
            description=description,
        )

        context['success'] = True  # To show success message in template

    return HttpResponse(template.render(context, request))
