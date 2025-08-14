# Python base image
FROM python:3.10-slim

# Work directory
WORKDIR /app

# Update pip and install build dependencies
RUN pip install --upgrade pip

# Requirements install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Streamlit port expose
EXPOSE 8501

# Run command
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

