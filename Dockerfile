FROM tensorflow/tensorflow:2.19.0-gpu

WORKDIR /app

# Speed up installs and set HF cache
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

# Copy current repo into image
COPY . /app

# Python deps
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt \
 && pip install -r cicd-requirements.txt \
 && pip install mlflow

# Default: show script help; supply args at run-time
CMD ["python", "phishing_email_detection_gpt2.py", "--help"]
