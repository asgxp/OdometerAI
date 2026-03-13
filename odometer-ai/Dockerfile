# -----------------------------
# Base Image
# -----------------------------
FROM python:3.10-slim

# -----------------------------
# Environment
# -----------------------------
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# -----------------------------
# Install system deps
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Workdir
# -----------------------------
WORKDIR /sdms/ai-odometer

# -----------------------------
# Install python deps (LOCKED)
# -----------------------------
COPY requirements.lock.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.lock.txt

# -----------------------------
# Copy source code
# -----------------------------
COPY . .


# -----------------------------
# Keep default models
# -----------------------------
RUN mkdir -p /sdms/ai-odometer/default_models \
    && cp -r /sdms/ai-odometer/models/* /sdms/ai-odometer/default_models/

# -----------------------------
# Entrypoint
# -----------------------------
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]


# -----------------------------
# Run app
# -----------------------------
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

