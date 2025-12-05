FROM python:3.9-slim
WORKDIR /app 
# Install system dependencies 
RUN apt-get update && apt-get install -y \ build-essential \ && rm -rf /var/lib/apt/lists/* 
# Copy requirements
COPY trainer/requirements.txt . 
# Install Python dependencies 
RUN pip install --no-cache-dir -r requirements.txt 
# Copy training code 
COPY trainer/ ./trainer/ 
# Set entrypoint 
ENTRYPOINT ["python", "-m", "trainer.train"]
