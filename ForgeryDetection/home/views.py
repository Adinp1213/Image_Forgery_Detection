from django.shortcuts import render,HttpResponse,redirect,get_object_or_404 
from home.models import UserDetails,Detections
from django.contrib import messages
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('forgery_discriminator_model.h5')
# Create your views here.
def index(request):
    return render(request,'index.html')

def login(request):
    return render(request,'login.html')

def create(request):
    return render(request,'createaccount.html')

def logout(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')

        try:
            user = UserDetails.objects.get(username = username)
        except UserDetails.DoesNotExist:
            user = None
        
        if user is not None:
            if(user.password == password):
                messages.success(request, 'You have successfully logged in.')
                request.session['user_id'] = user.id
                request.session['username'] = user.username
                
                return redirect("/")
            # Passwords match, redirect to home
            else:
                messages.error(request, 'Wrong password')
                return redirect("/login")

        else:
            messages.error(request, 'Account not found')
            return redirect("/login")

    return render(request, 'logout.html')

def failed(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        email = request.POST.get('email')
        phone= request.POST.get('phone')
        confirmPassword = request.POST.get('confirmPassword')

        if UserDetails.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists. Please choose a different one.')
            return redirect("/create") 
        
        if password!=confirmPassword:
            return render(request, 'failed.html')
        
        else:
            user = UserDetails(username = username, password = password,email =email,phone=phone)
            user.save()
            messages.success(request, 'Account created successfully!')
            return redirect("/") 

    return render(request, 'failed.html')

def logout_2(request):
    request.session.clear()
    return redirect('/')

def image_forgery(request):
    user_id = request.session.get('user_id')
    if user_id:
        return render(request, 'image_forgery.html')
    else:
        return redirect("/login") 
    
def detection_result(request):
    
    if request.method == "POST":
        detection_image = request.FILES['detection_image']
        img = Image.open(detection_image).resize((128, 128))  # Resize to match model input size
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        threshold = 0.5
        prediction_label = "Real" if predictions[0, 0] > threshold else "Forged"
        prediction = prediction_label
        print(prediction)
        user_id = request.session.get('user_id')
        detection = Detections(detection_image = detection_image,prediction = prediction,user_id = user_id)
        detection.save()
        return render(request, 'detection_result.html',{'data': detection})
    
    else:
        return redirect("/image_forgery")
    
def my_predictions(request):
    user_id = request.session.get('user_id')
    if user_id:
        user_detections = Detections.objects.filter(user_id=user_id)
        return render(request, 'my_detections.html',{'data': user_detections})
    else:
        return redirect("/login")

def delete(request):
    id = request.GET.get('id')
    user_id = request.session.get('user_id')
    entry = get_object_or_404(Detections,id = id)
    entry.delete()
    return redirect('/my_predictions')