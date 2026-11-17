FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    PORT=8000

WORKDIR /app

COPY pyproject.toml uv.lock ./
# Extra dependency groups, e.g. `--group azure` for the optional Azure adapters.
# docker-compose.yml's evals service passes `--group evals`.
ARG UV_SYNC_ARGS=""
RUN uv sync --frozen --no-dev --no-install-project ${UV_SYNC_ARGS}

COPY brain ./brain
COPY providers ./providers
COPY app ./app
COPY frontend ./frontend
COPY skills ./skills

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
