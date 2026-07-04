# Minimal, reproducible CPU image for the fast (fallback) path.
# Heavy extras (torch, openmm, botorch) install via extras when a GPU is present.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip && pip install -e ".[dev]"

COPY . .

CMD ["python", "-c", "import sonoforge; print('SonoForge', sonoforge.__version__)"]
