FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for geospatial libraries
RUN apt-get update && apt-get install -y \
    gcc g++ \
    libgdal-dev libproj-dev libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create output directory
RUN mkdir -p output

EXPOSE 8001

ENV PYTHONPATH=/app
ENV PORT=8081

CMD ["python", "main.py", "--api", "--port", "8081"]