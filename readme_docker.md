Build image with

docker build .\docker\ -t ti_slam

docker build --progress=plain .\docker\ -t ti_slam

Run image with

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v .\:/ti-slam ti_slam

docker run --gpus all --ipc=host -it -v .\:/ti-slam ti_slam