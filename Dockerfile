# # Use an official Python image as the base
# FROM python:3.10

# # Set a working directory
# WORKDIR /app

# # # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*
# COPY . /app
# # Install Python dependencies
# RUN pip install -r requirements.txt

# # Copy the application code
# COPY . .

# # Expose the API port
# EXPOSE 8000

# # Start the API server
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
