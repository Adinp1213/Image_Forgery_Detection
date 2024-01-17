from django.contrib import admin
from django.urls import path,include
from home import urls
from home import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name = "home"),
    path("login/", views.login , name = "login"),
    path("create/", views.create , name = "create"),
    path("failed/", views.failed , name = "failed"),
    path("logout/", views.logout , name = "logout"),
    path("logout_2/", views.logout_2 , name = "logout_2"),
    path("image_forgery/", views.image_forgery , name = "image_forgery"),
    path("detection_result/", views.detection_result , name = "detection_result"),
    path("my_predictions/", views.my_predictions , name = "my_predictions"),
    path("delete/", views.delete , name = "delete"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)