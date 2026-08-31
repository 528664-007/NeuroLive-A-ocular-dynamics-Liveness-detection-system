# CPU image by default — training real workloads should be run natively on
# your RTX 3050 with CUDA-enabled torch (this image intentionally doesn't
# assume nvidia-docker is set up; swap the base image for
# nvidia/cuda:12.x-devel + `pip install torch --index-url .../cu121` if you
# want GPU training inside the container).
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# mamba-ssm needs a CUDA build toolchain not present in this CPU image —
# strip it here; Phase 2 uses its documented fallback backbone instead.
RUN sed -i '/mamba-ssm/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

CMD ["pytest", "tests/", "-v"]
