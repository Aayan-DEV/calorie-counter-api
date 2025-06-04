FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Railway will override this)
EXPOSE 8000

# Use the start script for proper Railway deployment
CMD ["bash", "start.sh"]