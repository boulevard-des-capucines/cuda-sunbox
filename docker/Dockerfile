ARG CUDA=12.2.0
ARG UBUNTU=22.04

FROM nvidia/cuda:${CUDA}-devel-ubuntu${UBUNTU}

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt-get update \
    && apt-get install -y  \
      build-essential \
      ssh \
      libssl-dev \
      wget \
      rsync \
      git \
    && apt-get clean

# install cmake
RUN export CMAKE_VER_TEMP=3.24.4 \
    && cd / \
    && mkdir cmake-installs \
    && cd cmake-installs \
    && wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER_TEMP}/cmake-${CMAKE_VER_TEMP}.tar.gz \
    && tar -zxvf cmake-${CMAKE_VER_TEMP}.tar.gz \
    && cd cmake-${CMAKE_VER_TEMP} \
    && ./bootstrap --prefix=/usr \
    && make \
    && make install \
    && cd / \
    && rm -R cmake-installs \
    && unset CMAKE_VER_TEMP \
    && rm -rf /var/lib/apt/lists/*
    
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        freeglut3-dev \
        libx11-dev \
        libxmu-dev \
        libxi-dev \
        libglu1-mesa \
        libglu1-mesa-dev \
        x11-utils \
        xterm \
    && git clone https://github.com/NVIDIA/cuda-samples.git \
    && cd cuda-samples \
    && make TARGET_ARCH=x86_64 dbg=1 SMS="75" HOST_COMPILER=g++ \
    && cd /
ENV NVCUDASAMPLES_ROOT=/cuda-samples/Common/
RUN export CPATH=$CPATH:/cuda-samples/Common/

# user
RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_clion \
  && mkdir /run/sshd

RUN useradd -m user \
  && yes password | passwd user

# entrypoint
CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_clion"]

# CUDA_VER=12.2.0 && UBUNTU_VER=20.04 && docker build -t cuda-${CUDA_VER}-ubuntu${UBUNTU_VER}-dev-img . --build-arg CUDA=${CUDA_VER} --build-arg UBUNTU=${UBUNTU_VER}
# docker run -d --rm --gpus all -p 127.0.0.1:2222:22 -p 127.0.0.1:7777:7777 --cap-add SYS_PTRACE --security-opt seccomp=unconfined --name cuda-12.2.0-dev-22.04 cuda-12.2.0-ubuntu20.04-dev-img 
