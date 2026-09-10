FROM mambaorg/micromamba:1.5.8

WORKDIR /app

USER root

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    libgomp1 \
    libxml2-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libcairo2-dev \
    libpango1.0-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY docker/environment-viz-env.yml /tmp/environment.yml

RUN micromamba env create -f /tmp/environment.yml && \
    micromamba clean --all --yes

RUN mkdir -p /opt/omiclass-docker && \
    micromamba list -n omiclass_env --json > /opt/omiclass-docker/conda-packages.json

ENV PYTHONPATH=/app
ENV PATH=/opt/conda/envs/omiclass_env/bin:$PATH
ENV CONDA_DEFAULT_ENV=omiclass_env
ENV R_HOME=/opt/conda/envs/omiclass_env/lib/R
ENV HOME=/tmp
ENV XDG_CACHE_HOME=/tmp/.cache
ENV MPLCONFIGDIR=/tmp/.cache/matplotlib
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Copy the current framework source from this build context.  The Docker image
# deliberately does not install or retrieve a historical OmiClass checkout.
COPY api.py /app/api.py
COPY omiclass /app/omiclass
COPY docker/runtime_provenance.py /opt/omiclass-docker/runtime_provenance.py
COPY docker/entrypoint.sh /opt/omiclass-docker/entrypoint.sh

# Fail the image build immediately if the current OmiClass import graph is
# missing a runtime dependency; this does not execute the pipeline.
RUN /opt/conda/envs/omiclass_env/bin/python -c \
    "import requests, statsmodels; from omiclass.pipeline.runner import run_pipeline"

# Preserve a checksum manifest so every result can be tied to the source that
# was actually copied into the image, including an uncommitted working tree.
RUN find /app/api.py /app/omiclass -type f \
      \( -name '*.py' -o -name '*.yaml' -o -name '*.yml' -o -name '*.json' -o -name '*.R' -o -name '*.r' -o -name '*.csv' \) \
      -print0 | sort -z | xargs -0 sha256sum > /opt/omiclass-docker/source.sha256 && \
    chmod 755 /opt/omiclass-docker/entrypoint.sh

RUN mkdir -p /tmp/.cache/mamba/proc && chmod -R 777 /tmp/.cache

ENTRYPOINT ["/opt/omiclass-docker/entrypoint.sh"]
