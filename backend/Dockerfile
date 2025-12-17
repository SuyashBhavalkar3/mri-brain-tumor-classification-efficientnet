FROM python:3.10-bullseye

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# TensorFlow thread limits (CRITICAL)
ENV TF_NUM_INTEROP_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1
ENV OMP_NUM_THREADS=1

WORKDIR /app

# Required runtime libs for TensorFlow + PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", \
    "-k", "uvicorn.workers.UvicornWorker", \
    "main:app", \
    "--bind", "0.0.0.0:8000", \
    "--workers", "1", \
    "--timeout", "120"]
