# fastapi_app/Dockerfile
FROM python:3.11-slim

# lib for tesseract
RUN apt-get update && apt-get install -y g++ autoconf automake libtool \
    autoconf-archive zlib1g-dev libicu-dev libpango1.0-dev libcairo2-dev

# install tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    libtesseract-dev libleptonica-dev tesseract-ocr tesseract-ocr-all

# install poppler
RUN apt-get update && apt-get install -y poppler-utils

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/backend
#uncomment for production
WORKDIR /app/backend

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003", "--reload"]
