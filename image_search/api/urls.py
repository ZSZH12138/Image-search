from django.urls import path
from . import views

urlpatterns=[
    path('',views.index,name="index"),
    path("register/",views.register,name="register"),
    path("login/",views.login_view,name="login"),
    path("main_page/",views.main_page,name="main_page"),
    path("logout/", views.logout_view, name="logout_view"),
    path("upload_image/",views.upload_image,name="upload_image"),
    path("delete_account/",views.delete_account,name="delete_account"),
    path("example/",views.example,name="example"),
    path("history/", views.history_view, name="history"),

]