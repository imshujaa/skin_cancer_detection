# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install the necessary libraries for OpenCV and libgthread
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Set the command to run your application
CMD ["python", "main.py"]
