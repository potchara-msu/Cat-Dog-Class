# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Copy requirements.txt to the working directory
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

ARG CACHEBUST=1

# Copy the rest of the application code
COPY . .

# Expose the port for the Flask app (adjust as needed)
EXPOSE 8080

# Command to run the Flask app
CMD ["python", "main.py"]
