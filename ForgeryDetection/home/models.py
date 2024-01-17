from django.db import models


# Create your models here.
class UserDetails(models.Model):
    username = models.CharField(max_length=122)
    email = models.CharField(max_length=122)
    password = models.CharField(max_length=122)
    phone = models.CharField(max_length=13)

class Detections(models.Model):
    detection_image = models.ImageField(upload_to='dectection_images/')
    prediction = models.CharField(max_length=20)
    user_id = models.CharField(max_length=100)