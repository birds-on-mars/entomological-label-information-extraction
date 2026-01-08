FROM python:3.10.14-bullseye AS base

WORKDIR /app
ENV PYTHONPATH=/app

# Copy entire project
COPY . /app

# Common system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libstdc++6 \
    libjpeg-dev \
    libpng-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# SEGMENTATION STAGE
# ============================================
FROM base AS segmentation

RUN apt-get update && apt-get install -y libomp5 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r pipelines/requirements/segmentation.txt

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

CMD ["python", "scripts/processing/detection.py"]

# ============================================
# ROTATION STAGE
# ============================================
FROM base AS rotation

RUN pip install --no-cache-dir -r pipelines/requirements/rotation.txt
CMD ["python", "scripts/processing/rotation.py"]

# ============================================
# CLASSIFICATION STAGE
# ============================================
FROM base AS classification

RUN apt-get update && apt-get install -y python3-opencv && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r pipelines/requirements/classifier.txt

CMD ["python", "scripts/processing/classifiers.py"]

# ============================================
# TESSERACT STAGE
# ============================================
FROM base AS tesseract

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r pipelines/requirements/tesseract.txt
CMD ["python", "scripts/processing/tesseract.py"]

# ============================================
# EMPTY LABELS STAGE
# ============================================
FROM base AS empty_labels

RUN pip install --no-cache-dir -r pipelines/requirements/empty_labels.txt
CMD ["python", "scripts/processing/empty_labels.py"]

# ============================================
# POSTPROCESSING STAGE
# ============================================
FROM base AS postprocessing

RUN pip install --no-cache-dir -r pipelines/requirements/postprocess.txt
RUN python -c 'import nltk; nltk.download("punkt"); nltk.download("punkt_tab");'

CMD ["python", "scripts/postprocessing/process.py"]
