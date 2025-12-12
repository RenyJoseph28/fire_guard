from django.shortcuts import render

# Create your views here.

# Home view
def home(request):
    return render(request, 'public/index.html')

def signin(request):
    return render(request, 'public/signin.html')

def signup(request):
    return render(request, 'public/signup.html')
