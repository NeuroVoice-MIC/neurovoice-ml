FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps required for soundfile/librosa
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Force wheels only
RUN pip install --upgrade pip \
 && pip install --only-binary=:all: -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
