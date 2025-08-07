# Use an official Python runtime
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the code
COPY . .

# Expose Streamlit port
EXPOSE 8080

# Set environment variable for Streamlit
ENV PORT 8080

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
