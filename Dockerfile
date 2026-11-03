# syntax=docker/dockerfile:1
# The inspection pipeline, headless: Ubuntu 22.04 and Python 3.10 as on the CI runner, MuJoCo
# rendering on Mesa's software EGL device, the versions pinned in constraints.txt, run as a non-root
# user. PyTorch (the `learn` extra) is not included.
#
#   docker build -t auvinspect .
#       the navigation package is built from its repository at the commit pinned in pyproject.toml
#   docker build -t auvinspect --build-context nav=<directory holding its wheel> .
#       ... or taken from a wheel already built
#   docker run --rm -v "$PWD/runs:/work/runs" auvinspect benchmark run --suite ci --out runs/bench/ci

ARG BASE=ubuntu:22.04@sha256:076e7bf82319619f28dd0aa9c02ab34b055ad37100395b4f49a2ffba04fb5e11

# The navigation package's wheel, from its repository at the pinned commit (replaced by --build-context nav=...)
FROM ${BASE} AS nav-build
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git python3-pip \
 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml /tmp/pyproject.toml
RUN pip3 wheel --no-deps --wheel-dir /wheels "$(grep -o 'auv-navigation-fusion @ git+[^"]*' /tmp/pyproject.toml)"

FROM scratch AS nav
COPY --from=nav-build /wheels/ /

FROM ${BASE}
ENV DEBIAN_FRONTEND=noninteractive MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip libegl1 libopengl0 libegl-mesa0 libgl1-mesa-dri \
 && rm -rf /var/lib/apt/lists/* \
 && python3.10 -m pip install --no-cache-dir uv==0.13.1
WORKDIR /opt/auvinspect
COPY pyproject.toml constraints.txt README.md ./
COPY src ./src
COPY tests ./tests
COPY --from=nav / /tmp/wheels/
# The navigation requirement is overridden by the wheel, so nothing is fetched from a repository
# here; PyTorch lines are dropped from the constraints (CPU builds come from their own index).
RUN grep -vE '^(torch|torchvision)==' constraints.txt > /tmp/constraints.txt \
 && echo "auv-navigation-fusion @ file://$(ls /tmp/wheels/auv_navigation_fusion-*.whl)" > /tmp/overrides.txt \
 && uv venv /opt/venv --python /usr/bin/python3.10 \
 && uv pip install --python /opt/venv/bin/python --no-cache -c /tmp/constraints.txt --override /tmp/overrides.txt ".[dev]" \
 && rm -rf /tmp/wheels /tmp/constraints.txt /tmp/overrides.txt
ENV PATH=/opt/venv/bin:$PATH
RUN useradd --create-home --uid 1000 inspector && mkdir -p /work && chown inspector /work
USER inspector
WORKDIR /work
ENTRYPOINT ["auvinspect"]
CMD ["--help"]
