from django.http import HttpResponse
from django.template import loader
from . models import Contact
def homepage(request):
    template = loader.get_template("homepage.html")
    context = {}
    return HttpResponse(template.render(context, request))
def contactus(request):
    template = loader.get_template("contactus.html")
    context = {}

    if request.method == 'POST':
        first_name = request.POST.get('firstName', '').strip()
        last_name = request.POST.get('lastName', '').strip()
        email = request.POST.get('email', '').strip()
        phone_number = request.POST.get('phone', '').strip()
        subject = request.POST.get('subject', '').strip()
        description = request.POST.get('message', '').strip() 

        # Save to database
        Contact.objects.create(
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone_number=phone_number,
            subject=subject,
            description=description,
        )

        context['success'] = True

    return HttpResponse(template.render(context, request))
    

