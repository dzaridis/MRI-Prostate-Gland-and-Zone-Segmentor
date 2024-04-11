# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container to root
WORKDIR /

# Copy the current directory contents into the container at root
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Define mountable directories
VOLUME ["/pats", "/Outputs"]

# Run model.py when the container launches
CMD ["python", "./__main__.py"]