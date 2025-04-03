FROM python:3.11-slim

WORKDIR /app/frontend

# lib for tesseract
RUN apt-get update && apt-get install -y g++ autoconf automake libtool \
    autoconf-archive zlib1g-dev libicu-dev libpango1.0-dev libcairo2-dev

# install tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    libtesseract-dev libleptonica-dev tesseract-ocr tesseract-ocr-all

# install poppler
RUN apt-get update && apt-get install -y poppler-utils

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
EXPOSE 1234

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
