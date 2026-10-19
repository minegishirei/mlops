FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY flask_app.py ./
COPY templates ./templates

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "flask_app.py"]