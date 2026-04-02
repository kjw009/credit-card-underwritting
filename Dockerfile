FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

# instance/ is the SQLite data directory — mount a volume here in production
RUN mkdir -p instance

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
