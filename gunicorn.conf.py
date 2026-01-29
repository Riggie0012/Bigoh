import os

# Bind to Railway's injected PORT without relying on shell expansion.
port = os.getenv("PORT", "8000")
bind = f"0.0.0.0:{port}"

workers = int(os.getenv("WEB_CONCURRENCY", "2"))
threads = int(os.getenv("GUNICORN_THREADS", "1"))
timeout = int(os.getenv("GUNICORN_TIMEOUT", "30"))
