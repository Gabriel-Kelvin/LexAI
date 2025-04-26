from django.db import models
from django.contrib.auth.models import User


# Your existing Website model (assuming it looks something like this)
class Website(models.Model):
    name = models.CharField(max_length=100)
    url = models.URLField()

    def __str__(self):
        return self.name


# Your existing ChatMessage model
class ChatMessage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    website = models.ForeignKey(Website, on_delete=models.CASCADE)
    content = models.TextField()
    is_user = models.BooleanField(default=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.website.name} - {'User' if self.is_user else 'AI'}"


# New LegalChatMessage model for the Legal Assistant feature
class LegalChatMessage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()
    is_user = models.BooleanField(default=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"{self.user.username} - {'User' if self.is_user else 'AI'} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"


# Add to models.py
class Document(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


class DocumentChatMessage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, null=True, blank=True)
    is_user = models.BooleanField(default=True)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {'User' if self.is_user else 'AI'} - {self.timestamp}"