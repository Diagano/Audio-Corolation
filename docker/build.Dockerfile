# Rust toolchain for Linux and Windows (x86_64-pc-windows-gnu, statically linked CRT) builds.
# Build:  docker build -f docker/build.Dockerfile -t audio-offset-build docker
# Run:    docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/src" -v audio-offset-cargo:/cargo audio-offset-build \
#             cargo build --release --target x86_64-pc-windows-gnu
# The named volume caches downloaded crates; it is initialised from /cargo (world-writable) so any UID can use it.
FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl build-essential pkg-config mingw-w64 \
       libx11-dev libxcursor-dev libxrandr-dev libxi-dev libxkbcommon-dev libwayland-dev libgl-dev libasound2-dev \
    && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/opt/rustup \
    PATH=/opt/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | CARGO_HOME=/opt/cargo sh -s -- -y --no-modify-path --profile minimal \
        --default-toolchain stable --component clippy,rustfmt --target x86_64-pc-windows-gnu \
    && mkdir -m 1777 /cargo
ENV CARGO_HOME=/cargo \
    CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUSTFLAGS="-C target-feature=+crt-static"
WORKDIR /src
