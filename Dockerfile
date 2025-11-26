# 1. Base Image: Use a lightweight Python version
FROM python:3.10-slim

# 2. Set Working Directory inside the container
WORKDIR /app

# 3. Copy Dependencies First (Better Docker Caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the application code
COPY . .

# 5. Expose Port 8000 for the API
EXPOSE 8000

# 6. Start Command (Run the FastAPI app)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]