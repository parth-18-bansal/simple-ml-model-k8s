# Use an official Python runtime as base image
FROM python:3.12.3-slim

# Set working directory
WORKDIR /app

# Copy files
COPY app.py mnist_model.h5 train_model.py /app/

# Install dependencies
RUN pip install --no-cache-dir flask tensorflow numpy

# Expose the Flask port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
