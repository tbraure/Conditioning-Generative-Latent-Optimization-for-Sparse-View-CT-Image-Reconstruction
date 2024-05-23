# base nvidia CUDA image
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# install important dependencies
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt -y upgrade && \
    apt -y install apt-utils && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt update && \
    apt install python3.8 -y && \
    apt install -y python-is-python3 && \
    apt install -y python3-pip && \
    apt install -y git && \
    apt install -y build-essential && \
    apt install -y autotools-dev automake libtool && \
    apt install -y libcudnn8 && \
    apt install -y curl && \
    apt install -y libboost-all-dev && \
    apt install -y vim

RUN pip install -U numpy==1.19.5 scipy==1.5.2 pylidc==0.2.3 pydicom==2.4.0 \
                --no-cache-dir

RUN pip install -U torch==1.10.2+cu111 torchvision==0.11.3+cu111 \
                -f https://download.pytorch.org/whl/torch_stable.html \ 
                --no-cache-dir

## symlink
RUN ln -s /usr/local/cuda/lib64/libcusolver.so.11 /usr/local/cuda/lib64/libcusolver.so.10
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"

COPY . /src/
RUN cd /src/ && python setup.py install

