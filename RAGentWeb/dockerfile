# Builder Stage
FROM python:3.12-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install torch (CPU)
RUN pip install --no-cache-dir \
    torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download embedding model
RUN mkdir -p models/embeddings && \
    curl -L -o models/embeddings/pytorch_model.bin https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2/resolve/main/pytorch_model.bin && \
    curl -L -o models/embeddings/config.json https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2/resolve/main/config.json && \
    curl -L -o models/embeddings/tokenizer.json https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2/resolve/main/tokenizer.json && \
    curl -L -o models/embeddings/vocab.txt https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2/resolve/main/vocab.txt && \
    curl -L -o models/embeddings/special_tokens_map.json https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2/resolve/main/special_tokens_map.json

# Runtime Stage
FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libjpeg62-turbo zlib1g curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy libraries and embeddings from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/models/embeddings models/embeddings

# Create user first
RUN useradd -m -s /bin/bash appuser

# Copy application files
COPY src/ src/
COPY static/ static/
COPY templates/ templates/
COPY requirements.txt .
COPY data/vector_store/ data/vector_store/
COPY models/ml/ models/ml/
COPY models/dl/ models/dl/
COPY models/class_labels.txt models/class_labels.txt

# Copy the large H5 model file with proper ownership
COPY models/resnet_model.h5 models/resnet_model.h5

# Verify model file was copied correctly
RUN test -f models/resnet_model.h5 && \
    test -s models/resnet_model.h5 && \
    echo "Model file exists and is not empty" || \
    (echo "ERROR: resnet_model.h5 not copied or is empty!" && exit 1)

# Verify class_labels.txt exists
RUN test -f models/class_labels.txt && \
    echo "Labels file exists" || \
    (echo "ERROR: class_labels.txt not found!" && exit 1)

# Change ownership of all files to appuser
RUN chown -R appuser:appuser /app

# Environment variables
ENV PATH=/usr/local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    FAISS_PATH=/app/data/vector_store/faiss_index \
    PYTHONPATH=/app

USER appuser

EXPOSE 8000

# Final verification as appuser
RUN ls -lh models/ && \
    test -r models/resnet_model.h5 && \
    test -r models/class_labels.txt && \
    echo "Model files verified and readable" || \
    (echo "WARNING: Model files may not be readable" && true)

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
