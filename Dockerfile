FROM python:3.11-slim

LABEL maintainer="lexcrisis"
LABEL org.opencontainers.image.title="LexCrisis Legal Environment"
LABEL org.opencontainers.image.description="OpenEnv environment for multi-dimensional legal crisis management"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
