FROM python:3.8

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libhdf5-dev \
    libhdf5-serial-dev \
    hdf5-tools \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY final_model/ /app/final_model/
COPY label_encoder.joblib /app/

EXPOSE 8080

ENV PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]