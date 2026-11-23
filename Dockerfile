# UNTESTED in the build sandbox (no Docker available). CPU-only torch keeps the image smaller.
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .
COPY sql ./sql
COPY data ./data
COPY scripts ./scripts
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "telemetry.consume.decoder_consumer"]
