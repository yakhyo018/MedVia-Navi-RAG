FROM python:3.11-slim

# No .pyc files, logs straight to stdout (docker logs).
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Not root: a hole in the app does not hand over the container.
RUN useradd --create-home navi
USER navi

EXPOSE 8100
CMD ["uvicorn", "navi.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8100"]
