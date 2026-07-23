# docker/Dockerfile
# Multi-stage build:
#   Stage 1 (builder): installs all Python deps with uv into /app/.venv
#   Stage 2 (runtime): copies only the venv and source — no build tools, no uv

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (Docker layer caching: only re-runs if these change)
COPY pyproject.toml uv.lock* ./

# Install production dependencies only (no dev extras like pytest, jupyter)
# --no-install-project: don't install the package itself yet (we copy src later)
RUN uv sync --frozen --no-dev --no-install-project

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install OpenMP — required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Create non-root user for security (never run as root in production)
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# Copy the installed virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/

# Set PATH to use the venv Python
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

# Expose the FastAPI port
EXPOSE 8000

# Health check — Docker will mark container unhealthy if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Run the FastAPI app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
