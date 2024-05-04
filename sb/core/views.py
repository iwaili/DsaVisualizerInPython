from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib import messages
from django.contrib.auth.models import User, auth
from .models import Profile

import re 
def index(request):
    return render(request,'hello.html')

def signup(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        confirmpassword = request.POST['confirmPassword']

        if password == confirmpassword:
            if User.objects.filter(username=username).exists():
                messages.info(request,'Username already taken')
                return redirect('signup')
            else:
                check=0
                if len(password) < 8:
                    check=1
                # Check if password contains at least one digit
                elif not re.search(r'\d', password):
                    check=1
                # Check if password contains at least one special character
                elif not re.search(r'[!@#$%^&*()-_+=]', password):
                    check=1
                if check==1:
                    messages.info(request,'Password is not strong enough ')
                    return redirect('signup')
                else:
                    user = User.objects.create_user(username=username,password=password)
                    user.save( )
                    user_model = User.objects.get(username = username)#profile belongs to which user
                    new_profile = Profile.objects.create(user=user_model,id_user=user_model.id)
                    new_profile.save()
                    messages.info(request,'Signup Complete')
                    return redirect('signin')
                

        else:
            messages.info(request,'Passwords not matching')
            return redirect('signup')
    else: 
        return render(request,'signup.html')

def signin(request):
    return render(request,'signin.html')
# Create your views here.
