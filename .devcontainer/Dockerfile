FROM mcr.microsoft.com/devcontainers/anaconda:0-3

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

ENV MY_DIR=/root/richards
ENV TZ=Europe/Rome

WORKDIR ${MY_DIR}

RUN apt update && apt install python3-dev python3-pip gmsh git metis curl -y

# Python requirements
RUN python3 -m pip install cython numba shapely ipython ipykernel jupyterlab

# Porepy
COPY porepy porepy
RUN python3 -m pip install -r porepy/requirements.txt
RUN python3 -m pip install -r porepy/requirements-dev.txt

WORKDIR ${MY_DIR}/porepy
RUN python3 -m pip install .

WORKDIR ${MY_DIR}

# Pygeon
COPY pygeon pygeon
RUN python3 -m pip install -r pygeon/requirements.txt
RUN python3 -m pip install -r pygeon/requirements-dev.txt

WORKDIR ${MY_DIR}/pygeon
RUN python3 -m pip install .

WORKDIR ${MY_DIR}

COPY final_tests final_tests
COPY plotters plotters
COPY richards richards
COPY utilities utilities

ENV PYTHONPATH "${PYTHONPATH}:/workspaces/richards:/root/richards"

CMD bash