FROM python:3.11-slim

WORKDIR /app/frontend

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=8504", "--server.address=0.0.0.0"]
