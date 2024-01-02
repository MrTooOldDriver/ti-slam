Build image with

docker build .\docker\ 

Run image with

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v .\:/ti-slam ti_slam