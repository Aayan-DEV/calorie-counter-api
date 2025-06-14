FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
# Add before installing Python packages
RUN apt-get update && apt-get install -y \
    libzbar0 \
    libzbar-dev \
    && rm -rf /var/lib/apt/lists/*
    
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Use environment variable for port with fallback
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]