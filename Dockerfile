FROM python:3.10-slim

WORKDIR /app

COPY app.py .

RUN pip install fastapi uvicorn scikit-learn requests prometheus-client

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
