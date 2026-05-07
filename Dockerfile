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

COPY environment.yml /tmp/environment.yml

RUN micromamba env create -f /tmp/environment.yml && \
    micromamba clean --all --yes

COPY . /app

ENV PYTHONPATH=/app
ENV PATH=/opt/conda/envs/omiclass_env/bin:$PATH
ENV CONDA_DEFAULT_ENV=omiclass_env
ENV HOME=/tmp
ENV XDG_CACHE_HOME=/tmp/.cache

RUN mkdir -p /tmp/.cache/mamba/proc && chmod -R 777 /tmp/.cache

ENTRYPOINT ["micromamba", "run", "-n", "omiclass_env", "python", "api.py"]