FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
RUN pip install --no-cache-dir numpy
COPY reference.py /app/reference.py
COPY kb_verifier.py /app/kb_verifier.py
COPY config.json /app/config.json
WORKDIR /app
