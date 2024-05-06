from django.urls import path
from . import views
urlpatterns = [
    path('',views.index,name ='index'),
    path('index',views.index,name='index'),
    path('signup',views.signup,name = 'signup'),
    path('signin',views.signin,name='signin'),
    path('about',views.about,name='about'),
    path('logout',views.logout,name='logout'),
    path('actual',views.actual,name='actual'),    
]