FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first (leverages Docker layer caching)
COPY trainer/requirements.txt .
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy the training package
COPY trainer/ ./trainer/
# Set entrypoint for training
ENTRYPOINT ["python", "-m", "trainer.train"]

