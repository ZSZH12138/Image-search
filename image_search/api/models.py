from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class SearchHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    query_image = models.ImageField(upload_to="history/")
    top_k = models.IntegerField(default=10)
    results = models.JSONField()  # 存 img_url + score + description
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} @ {self.created_at}"