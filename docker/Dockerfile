FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3

ENV TZ=Europe/London

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-pil \
    python3-pil.imagetk \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg

RUN python -m pip install --upgrade pip

RUN pip3 install --no-cache-dir \
    tensorflow==2.3.1 \
    keras==2.4.0 \
    scipy \
    matplotlib \
    opencv-python

RUN pip3 install --no-cache-dir \
    flwr[simulation] \
    flwr-datasets[vision] \
    tensorflow-probability==0.7 \
    keras-mdn-layer

# tensorflow-probability==0.7 might have issue?

RUN pip3 install --no-cache-dir \
    numpy==1.21.0

RUN