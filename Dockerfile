# Use official Python slim image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Install uv
RUN pip install uv

# Copy requirements first (so Docker can cache this layer)
COPY requirements.txt .

# Install dependencies using uv into system Python (no venv needed inside container)
RUN uv pip install --system -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
