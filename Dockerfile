FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

ENV PATH="/app/.venv/bin:$PATH"
ENV DATABASE_URL="postgresql+psycopg://build:build@localhost:5432/build"
ENV PORT=10000

# Build the static Chroma knowledge base into the image.
RUN uv run python -c "from retrieval.chroma_store import ingest_kb; ingest_kb()"

EXPOSE 10000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]