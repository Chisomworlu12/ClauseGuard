FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-group retrieval-heavy --no-group fine-tuning

COPY . .

ENV PATH="/app/.venv/bin:$PATH"
ENV DATABASE_URL="postgresql+psycopg://build:build@localhost:5432/build"
ENV PORT=10000


EXPOSE 10000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]