from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()
class Profile(models.Model):
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    id_user = models.IntegerField()
    profileimg = models.ImageField(upload_to='profile_images',default='default.png')#we want to save where we want store this image so we have media
    
    def __str__(self):
        return self.user.username


# Create your models here.
#if we make any changes here we need it to migrate it by writing python manage.py makemigrations
'''
Django models are Python classes that represent database tables.
They define the structure of the data that Django manages and provide an abstraction layer for interacting with the database.
Each model class corresponds to a table in the database, and each attribute of the class represents a field in that table.

'''