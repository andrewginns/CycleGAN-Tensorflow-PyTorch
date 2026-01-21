FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Minimal OS deps for common image libs
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
  && rm -rf /var/lib/apt/lists/*

# uv for fast, locked installs
RUN pip install --no-cache-dir uv

WORKDIR /app

# Install dependencies first for better layer caching
COPY pyproject.toml uv.lock ./
COPY cyclegan_torch ./cyclegan_torch

# Install repo deps from the lockfile (torch is intentionally excluded).
RUN uv sync --frozen

# Install CPU-only PyTorch wheels (avoids pulling CUDA runtimes).
RUN uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

ENV PATH="/app/.venv/bin:${PATH}"

CMD ["bash"]
