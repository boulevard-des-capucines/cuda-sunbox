ARG CUDA_VER=12.2.0
ARG UBUNTU_VER=22.04

FROM nvidia/cuda:${CUDA_VER}-devel-ubuntu${UBUNTU_VER}

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
ARG CMAKE_VER=3.26.4
RUN cd / \
    && mkdir cmake-installs \
    && cd cmake-installs \
    && wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER}/cmake-${CMAKE_VER}.tar.gz \
    && tar -zxvf cmake-${CMAKE_VER}.tar.gz \
    && cd cmake-${CMAKE_VER} \
    && ./bootstrap --prefix=/usr \
    && make \
    && make install \
    && cd / \
    && rm -R cmake-installs \
    && rm -rf /var/lib/apt/lists/*
    
# install cuda-samples and add path to ENVs
RUN export NV_CUDA_SAMPLES_INCLUDE=/cuda-samples/Common  \
    && apt-get update \
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
    && cd /  \
    && ( \
        echo "export NV_CUDA_SAMPLES_INCLUDE=${NV_CUDA_SAMPLES_INCLUDE}"; \
        echo "export CPATH=${CPATH}:${NV_CUDA_SAMPLES_INCLUDE}"; \
    ) | tee /etc/bash.bashrc > /etc/profile.d/export-cuda-samples-envs.sh

# user
RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_ide \
  && mkdir /run/sshd

RUN useradd -m -s /bin/bash user \
  && yes password | passwd user

# entrypoint
CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_ide"]

#https://github.com/diegoferigo/devenv/issues/2#issue-396393875
#https://stackoverflow.com/questions/30590083/how-do-i-rename-both-a-git-local-and-remote-branch-name

# CUDA_VER=12.2.0 && UBUNTU_VER=20.04 && CMAKE_VER=3.26.4 && sudo docker build -t cuda-${CUDA_VER}-ubuntu${UBUNTU_VER}-dev-img . --build-arg CUDA_VER=${CUDA_VER} --build-arg UBUNTU_VER=${UBUNTU_VER} --build-arg CMAKE_VER=${CMAKE_VER}
# sudo docker run -d --rm --gpus all -p 127.0.0.1:2222:22 -p 127.0.0.1:7777:7777 --cap-add SYS_PTRACE --security-opt seccomp=unconfined --name cuda-12.2.0-ubuntu22.04-dev cuda-12.2.0-ubuntu20.04-dev-img
