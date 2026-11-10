# CUDA driver comes from the host's NVIDIA Container Toolkit.
# Pick a tag compatible with your installed host driver; verify with nvidia-smi.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
ENV PYTHONPATH=/app/src
RUN pip install --no-cache-dir .
ENV FLEX_DEVICE=cuda:0
EXPOSE 8000
CMD ["uvicorn", "flexengine.serving.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
