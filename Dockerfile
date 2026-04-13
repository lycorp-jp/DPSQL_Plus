FROM ubuntu:22.04

WORKDIR /workspace/DPSQL

RUN apt-get update 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get update && apt-get install -y python3 python3-venv
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

RUN curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

COPY ./ /workspace/DPSQL/

RUN uv sync --locked
