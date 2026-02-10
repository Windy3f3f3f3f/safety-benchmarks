FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# China mirrors for faster builds
RUN sed -i 's|archive.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list && \
    sed -i 's|security.ubuntu.com|mirrors.aliyun.com|g' /etc/apt/sources.list
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# Vulkan + X11 libs required by AI2-THOR CloudRendering
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        libvulkan1 vulkan-tools mesa-vulkan-drivers \
        libgl1-mesa-glx libglib2.0-0 \
        xvfb wget unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY server.py low_level_controller.py ./

EXPOSE 9100
ENV PORT=9100

# Pre-create ai2thor directories. The actual binary is mounted via
# docker-compose volume (pre-downloaded on the host to avoid slow S3).
RUN mkdir -p /root/.ai2thor/releases /root/.ai2thor/tmp

# CloudRendering requires a virtual X display (Vulkan over Xvfb).
CMD bash -c 'Xvfb :99 -screen 0 1024x768x24 -nolisten tcp &  sleep 1 && DISPLAY=:99 exec python3 server.py'
