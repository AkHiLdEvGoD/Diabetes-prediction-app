FROM apache/airflow:3.1.5

USER airflow

# Install uv
RUN pip install --no-cache-dir uv

# Copy only pyproject (no code)
COPY pyproject.toml /opt/airflow/pyproject.toml

# Install ONLY project deps (Airflow already exists)
RUN uv pip install --no-deps \
    -r <(uv pip compile /opt/airflow/pyproject.toml)

