# syntax=docker/dockerfile:1
# Base image: official Python slim, pinned to a patch release (checked on Docker Hub, October 2026).
ARG PYTHON_IMAGE=python:3.12.15-slim-trixie

# --- build: resolve runtime dependencies into a virtualenv, pinned by the lockfile ---
FROM ${PYTHON_IMAGE} AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /app
COPY requirements.lock pyproject.toml ./
COPY src ./src
# The lockfile is used as constraints: only the project's runtime dependencies are installed, at locked versions
# (the lock also pins dev tools, which the image doesn't need). setuptools comes from the lock for the editable build.
# --no-compile and the __pycache__ cleanup keep bytecode out of the image (~120 MB) to stay closer to the size budget;
# the trade-off is a slower cold start, since Python compiles imported modules in memory on every start.
RUN pip install --no-cache-dir --no-compile -c requirements.lock pip setuptools \
 && pip install --no-cache-dir --no-compile -c requirements.lock --no-build-isolation -e . \
 && find /opt/venv -type d -name __pycache__ -prune -exec rm -rf {} + \
 # Drop the test suites bundled inside third-party packages (~140 MB); no runtime code imports them.
 && find /opt/venv/lib/python3.12/site-packages -type d \( -name tests -o -name test \) -prune -exec rm -rf {} +

# --- runtime ---
FROM ${PYTHON_IMAGE}
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH=/opt/venv/bin:$PATH
# libgomp1: the OpenMP runtime LightGBM loads at import (not in the slim base image).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
# Application files stay root-owned and read-only to the app user; only /app/data is writable.
COPY pyproject.toml requirements.lock ./
COPY src ./src
COPY .streamlit/config.toml ./.streamlit/config.toml
# The editable install points at /app/src, so the data directory resolves to /app/data (mount a volume there).
COPY --from=build /app/src/stockapp.egg-info ./src/stockapp.egg-info
RUN chmod -R a-w,a+rX /app/src /app/.streamlit /app/pyproject.toml /app/requirements.lock \
 && mkdir -p /app/data && chown app:app /app/data
VOLUME ["/app/data"]
USER app
EXPOSE 8501
# Streamlit 1.66 health endpoint (/_stcore/health); urllib because the slim image has no curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"]
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
