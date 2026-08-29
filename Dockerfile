FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.10-slim
ENV DEBIAN_FRONTEND=noninteractive
LABEL maintainer="hongyiyang <hongyiyangyhy@163.com>"


COPY . /root/project/

# 创建工作目录
WORKDIR /root/project/

RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources && \
    cd /root/project/ && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        tzdata \
        build-essential \
        python3-full \
        python3-dev \
        python3-pip \
        python3-yaml \
        ffmpeg \
        portaudio19-dev \
        libgtk-3-dev \
        git && \
\
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
pip install --break-system-packages --upgrade pip

RUN chmod +x /root/project/install/install_setup.sh
RUN mkdir -p /root/project/checkpoint
ENTRYPOINT ["/root/project/install/install_setup.sh"]
EXPOSE 8501
CMD ["streamlit", "run", "/root/project/gui.py"]
