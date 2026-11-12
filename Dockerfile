# Airflow image with our pipeline code.
#
# Two separate Python environments in one image:
# - Airflow's own (from the base image): runs scheduler, API server and the DAG files
# - ours in /opt/pipeline/.venv: runs the pipeline steps (polars, lightgbm, mlflow, ...)
# The tasks call our steps as commands (python -m db_delay_mlops...), so the two sets
# of packages never have to agree on versions.

FROM apache/airflow:3.3.2-python3.12

# LightGBM needs the OpenMP library (OSError: libgomp.so.1 without it)
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/pipeline && chown airflow:0 /opt/pipeline

# Same uv version as on the host, copied from its official image
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /usr/local/bin/uv

USER airflow
WORKDIR /opt/pipeline
ENV UV_PROJECT_ENVIRONMENT=/opt/pipeline/.venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# 1) Only the dependencies: this layer is cached as long as uv.lock does not change
COPY --chown=airflow:0 pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) Our code and config: changes often, so it comes last
COPY --chown=airflow:0 src ./src
COPY --chown=airflow:0 config ./config
RUN uv sync --frozen --no-dev

# data/ is not in the image: it is mounted as a volume (see docker-compose.yaml)
ENV PIPELINE_PYTHON=/opt/pipeline/.venv/bin/python
