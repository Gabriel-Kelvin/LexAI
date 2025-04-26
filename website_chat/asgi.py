"""
ASGI config for website_chat project.
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'website_chat.settings')

application = get_asgi_application()