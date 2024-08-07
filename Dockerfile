# python 3.9
FROM python:3.9.19

# Set the working directory
WORKDIR /app

ARG CACHEBUST=144

RUN git clone https://github.com/potchara-msu/Cat-Dog-Class.git .

# Copy the rest of the application code
COPY . .

# Copy requirements.txt to the working directory
# COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port for the Flask app (adjust as needed)
EXPOSE 8080

# Command to run the Flask app
CMD ["python", "main.py"]
