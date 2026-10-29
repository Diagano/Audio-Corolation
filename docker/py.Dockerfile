# Runtime for audio_offset.py: Python, numpy and ffmpeg from Ubuntu packages.
# Build:  docker build -f docker/py.Dockerfile -t audio-offset-py .
# Run:    docker run --rm -u "$(id -u):$(id -g)" -v /path/to/media:/media:ro audio-offset-py /media/a.mkv /media/b.mkv --mode landmarks
# Mount the working copy over /app (-v "$PWD:/app:ro") to run an edited script without rebuilding.
FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-numpy ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY audio_offset.py /app/audio_offset.py
WORKDIR /media
ENTRYPOINT ["python3", "/app/audio_offset.py"]
