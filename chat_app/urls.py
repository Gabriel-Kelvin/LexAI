from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('signup/', views.signup_view, name='signup'),
    path('chat/', views.chat_view, name='chat'),
    path('get_response/', views.get_ai_response, name='get_response'),
    # New URLs for Legal Assistant
    path('legal/', views.legal_assistant_view, name='legal_assistant'),
    path('get_legal_response/', views.get_legal_response, name='get_legal_response'),
    path('clear_legal_history/', views.clear_legal_history, name='clear_legal_history'),
    #New URLs for Document Analyzer
    path('documents/', views.document_analyzer_view, name='document_analyzer'),
    path('documents/upload/', views.upload_document, name='upload_document'),
    path('documents/delete/<int:document_id>/', views.delete_document, name='delete_document'),
    path('get_document_response/', views.get_document_response, name='get_document_response'),
    path('clear_document_history/', views.clear_document_history, name='clear_document_history'),
    ]