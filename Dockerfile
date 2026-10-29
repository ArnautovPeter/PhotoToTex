FROM ghcr.io/astral-sh/uv:0.12.23 AS uv
FROM python:3.11-slim
COPY --from=uv /uv /uvx /bin/
WORKDIR /app
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --extra cpu --no-dev
COPY src ./src
COPY licenses ./licenses
EXPOSE 8000
CMD ["uv", "run", "--frozen", "--extra", "cpu", "--no-dev", "python", "-m", "src.web_app", "--host", "0.0.0.0"]
