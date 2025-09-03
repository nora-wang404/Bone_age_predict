FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install pip==23.3.2 && \
    pip install -r requirements.txt


COPY . .

EXPOSE 5000
CMD ["python", "main.py"]
