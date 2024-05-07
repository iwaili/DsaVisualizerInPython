from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views
urlpatterns = [
    path('',views.index,name ='index'),
    path('index',views.index,name='index'),
    path('signup',views.signup,name = 'signup'),
    path('signin',views.signin,name='signin'),
    path('about',views.about,name='about'),
    path('signin/about',views.about,name='about'),
    path('logout',views.logout,name='logout'),
    path('actual',views.actual,name='actual'), 
    path('actual/process/', views.process_data, name='process_data'),
    path('show/<str:username>/<int:requestno>/', views.show, name='show'),
    path('actual/rulesForMatrix',views.rulesForMatrix,name='rulesForMatrix'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)