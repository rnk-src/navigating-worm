# A CPU-only Linux image that runs `make check` the same way on every machine.
# Build and check: make docker-check
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

# EGL and OSMesa let MuJoCo render headless. git, tmux and ps (procps) are for the tests.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        make git tmux procps libgl1 libegl1 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=egl \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install the locked dependencies first, so a code change does not reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

CMD ["make", "check"]
