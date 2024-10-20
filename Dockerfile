# Use a base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make the container's port 8080 available to the outside world
EXPOSE 8080

# Define environment variable
ENV NAME World

# Run the application
CMD ["python", "house_price_prediction.py"]
 
