FROM python:3.11-bookworm


RUN mkdir -p /app
WORKDIR /app/

COPY uqma uqma
COPY pyproject.toml .
COPY README.md .

RUN pip install .

ENTRYPOINT [ "python3", "-m", "uqma.scripts.main"]
