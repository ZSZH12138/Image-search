from django.contrib.auth import authenticate, login, logout
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.db import IntegrityError
from .dinov2_numpy import Dinov2Numpy
from .models import SearchHistory
from .preprocess_image import center_crop
import numpy as np
import os
# Create your views here.

def index(request):
    return render(request,"api/index.html")

def main_page(request):
    return render(request,"api/main_page.html")

def register(request):
    if request.method=="POST":
        u=request.POST.get("username")
        p=request.POST.get("password")
        cp=request.POST.get("confirm_password")
        if p != cp:
            return render(request, "api/register.html", {"error": "两次密码不一致"})
        try:
            User.objects.create_user(username=u, password=p)
            return login_view(request)
        except IntegrityError:
            return render(request, "api/register.html", {"error": "用户名已存在，请换一个"})
        except:
            return render(request, "api/register.html", {"error": "注册失败，请稍后再试"})
    return render(request,"api/register.html")


def login_view(request):
    if request.method == "POST":
        u = request.POST.get("username")
        p = request.POST.get("password")

        # 验证用户名和密码是否匹配
        user = authenticate(request, username=u, password=p)

        if user is not None:
            # 验证通过，正式建立会话 (Session)
            login(request, user)
            return redirect("main_page")  # 登录成功跳转到首页
        else:
            # 验证失败
            return render(request, "api/login.html", {"error": "用户名或密码错误"})

    return render(request, "api/login.html")

def logout_view(request):
    if request.method == 'POST':
        logout(request)
    return redirect('index')

def upload_image(request):
    if request.method=="POST" and request.FILES.get("image"):
        img=request.FILES["image"]
        fs=FileSystemStorage()
        filename=fs.save(img.name,img)
        img_path=fs.path(filename)
        img_url=fs.url(filename)
        return get_img_feature(request,img_path,img_url)
    return redirect("main_page")

def get_img_feature(request,img_path,img_url):
    weights = np.load("api/vit-dinov2-base.npz")
    vit = Dinov2Numpy(weights)
    pixel_value=center_crop(img_path)
    feat=vit(pixel_value)
    return show_result(request,feat,img_url)

def show_result(request, feat, img_url):
    db = np.load("api/data/features.npy", allow_pickle=True).item()
    query_feat = feat.reshape(-1)
    query_feat = query_feat / np.linalg.norm(query_feat)
    results = []
    for img_name, db_feat in db.items():
        db_feat = db_feat.reshape(-1)
        db_feat = db_feat / np.linalg.norm(db_feat)
        sim = float(np.dot(query_feat, db_feat))

        with open(f"api/data/database/{img_name.replace('.jpg', '.txt')}",
                  'r', encoding='utf-8') as f:
            description = f.read()
        results.append({
            "img_url": f"/media/database/{img_name}",
            "description": description.capitalize(),
            "score": sim
        })
    # 排序
    results.sort(key=lambda x: x["score"], reverse=True)
    top_k = request.POST.get("top_k", 10)  # 默认 10
    try:
        top_k = int(top_k)
    except ValueError:
        top_k = 10
    # 防止用户乱来
    top_k = max(1, min(top_k, 50))  # 1~50 之间
    results = results[:top_k]
    if request.user.is_authenticated:
        SearchHistory.objects.create(
            user=request.user,
            query_image=img_url,
            top_k=top_k,
            results=results
        )
    return render(request, "api/result.html", {
        "query_img": img_url,
        "results": results,
        "top_k": top_k
    })

def example(request):
    imgs_path="api/data/database"
    img_lst=os.listdir(imgs_path)
    res=""
    for i in img_lst:
        if i.endswith(".jpg"):
            res=i
            break
    return get_img_feature(request,os.path.join(imgs_path,res),f"/media/database/{res}")

def history_view(request):
    histories = SearchHistory.objects.filter(
        user=request.user
    ).order_by("-created_at")
    return render(request, "api/history.html", {
        "histories": histories
    })

def delete_account(request):
    if request.method == "POST":
        user = request.user
        logout(request)     # 先退出登录
        user.delete()       # 再删除账号
        return redirect("index")
    return render(request, "api/confirm_delete.html")
