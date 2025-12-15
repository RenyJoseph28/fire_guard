
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.hashers import make_password
from .models import user_registrations
from django.contrib.auth.hashers import check_password
from django.http import HttpResponse

# Create your views here.

# Home view
def home(request):
    return render(request, 'public/index.html')






def signup(request):
    if request.method == "POST":
        full_name = request.POST.get("full_name")
        email = request.POST.get("email")
        password1 = request.POST.get("password1")
        password2 = request.POST.get("password2")

        # Validate empty fields
        if not all([full_name, email, password1, password2]):
            messages.error(request, "All fields are required")
            return redirect("signup")

        # Validate passwords
        if password1 != password2:
            messages.error(request, "Passwords do not match")
            return redirect("signup")

        # Check if email exists
        if user_registrations.objects.filter(email=email).exists():
            messages.error(request, "Email already registered")
            return redirect("signup")

        # Hash password
        hashed_password = make_password(password1)
        print(hashed_password)

        # Save user
        user_registrations.objects.create(
            fullname=full_name,
            email=email,
            password=hashed_password,
            is_active=True
        )

        messages.success(request, "Account created successfully! Please sign in.")
        return redirect("signin")  # or wherever you want

    return render(request, "public/signup.html")







def signin(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")

        # Validate input
        if not email or not password:
            messages.error(request, "Email and password are required")
            return redirect("signin")

        try:
            user = user_registrations.objects.get(email=email)
        except user_registrations.DoesNotExist:
            messages.error(request, "Invalid email or password")
            return redirect("signin")

        # Check hashed password
        if not check_password(password, user.password):
            messages.error(request, "Invalid email or password")
            return redirect("signin")

        # Check active status
        if not user.is_active:
            messages.error(request, "Account is inactive")
            return redirect("signin")

        # Store session (manual login)
        request.session["user_id"] = user.id
        request.session["user_email"] = user.email
        request.session["user_name"] = user.fullname

        messages.success(request, "Login successful")
        return redirect("dashboard")  # change if needed

    return render(request, "public/signin.html")




def dashboard(request):
    return HttpResponse("""
        <h1>Welcome to User Dashboard</h1>
        <p>You are successfully logged in.</p>
    """)
