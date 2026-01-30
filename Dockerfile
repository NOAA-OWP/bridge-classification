FROM mambaorg/micromamba

# Set the working directory inside the container
WORKDIR /app

# Copy the environment file
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yaml /tmp/env.yaml

# Install dependencies into the base environment
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

# This ARG forces the container to activate the 'base' environment
# for all subsequent RUN and CMD instructions.
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Fix for the libstdc++ issue (prevents "version `CXXABI_1.3.15' not found" errors)
ENV LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH:-}"

# Copy the source code into the container
COPY --chown=$MAMBA_USER:$MAMBA_USER . .

# Default: interactive shell with env active (override for one-off commands)
CMD ["/bin/bash"]
