from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from .models import Signup

# Create your views here.
def signup(request):
    template = loader.get_template("signup.html")
    context = {}

    if request.method == 'POST':
        full_name = request.POST.get('fullName', '').strip()
        email_address = request.POST.get('email', '').strip()
        password = request.POST.get('password', '')
        confirm_password = request.POST.get('confirmPassword', '')
        if full_name and email_address and password and confirm_password and password == confirm_password:
            Signup.objects.create(
                full_name=full_name,
                email_address=email_address,
                password=password,
                confirm_password=confirm_password,
            )
            context['success'] = True
        else:
            context['error'] = 'Please complete the form correctly.'
            return HttpResponse(template.render(context, request))
