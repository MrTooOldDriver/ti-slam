FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3

ENV TZ=Europe/London

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-pil \
    python3-pil.imagetk \
    libsm6 \
    libxext6 \
    libxrender-dev

RUN pip3 install --no-cache-dir \
    keras==2.3.1 \
    scipy==1.1.0 \
    matplotlib \
    opencv-python

RUN pip3 install --no-cache-dir \
    flwr[simulation] \
    flwr-datasets[vision] \
    tensorflow-probability==0.7 \
    keras-mdn-layer==0.2.2 \
