from django.db import models
from django.contrib.auth import get_user_model
from datetime import date
User = get_user_model()
class Profile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    id_user = models.IntegerField()
    profileimg = models.ImageField(upload_to='profile_images', default='default.png')
    numrequests = models.IntegerField(default=0)
    
    def __str__(self):
        return self.user.username
    
    def num(self):
        return self.numrequests
    
    def incnum(self):
        self.numrequests += 1
    
    class Meta:
        permissions = [
            ('view_profile_info', 'Can view profile information'),
        ]


class userData(models.Model):
    name = models.CharField(max_length=200,default="harshit")
    input = models.TextField(default="error")  # Renaming 'input' field to 'input_data'
    which = models.CharField(max_length=100,default="dunno")  # Assuming 'which' is a string field
    date = models.DateField(default='2003-05-21')

    def __str__(self):
        return self.name 

class SavedGraph(models.Model):
    username = models.CharField(max_length=100)
    requestno = models.CharField(max_length=100)
    ab = models.CharField(max_length=100)
    image = models.ImageField(upload_to='graphs/')

    def __str__(self):
        return f"{self.username}_{self.requestno}_{self.ab}"

# Create your models here.
#if we make any changes here we need it to migrate it by writing python manage.py makemigrations
'''
Django models are Python classes that represent database tables.
They define the structure of the data that Django manages and provide an abstraction layer for interacting with the database.
Each model class corresponds to a table in the database, and each attribute of the class represents a field in that table.

'''