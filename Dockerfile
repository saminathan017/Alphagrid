FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y git curl awscli && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    lightgbm>=4.3.0 \
    scikit-learn>=1.4.0 \
    yfinance>=0.2.50 \
    pandas>=2.1.0 \
    numpy>=1.26.0 \
    loguru>=0.7.0 \
    scipy>=1.13.0 \
    pandas-ta>=0.3.14b \
    boto3>=1.34.0 \
    imbalanced-learn>=0.12.0

# Copy project files
COPY . .

# Default: full 10-year training
ENTRYPOINT ["python", "scripts/train_models.py"]
CMD ["--lookback", "3650"]
