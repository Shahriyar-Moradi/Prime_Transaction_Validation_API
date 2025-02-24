# Use Python as the base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
build-essential \
python3-dev \
libffi-dev \
libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*



# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy the .env file into the container
COPY .env .env

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main3:app", "--host", "0.0.0.0", "--port", "8000"]
