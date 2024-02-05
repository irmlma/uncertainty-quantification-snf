FROM python:3.11-bookworm


RUN mkdir -p /app
WORKDIR /app/

COPY uqma .
COPY pyproject.toml .

RUN pip install .


ENTRYPOINT [ "python3", "-m", "uqma.scripts.main"]
