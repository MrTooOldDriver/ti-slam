conda create -n ti-slam python=3.8

conda activate ti-slam

conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

pip3 install tensorflow==2.6.0 protobuf==3.20.0

TEST TENSORFLOW GPU INSTALLATION WITH THIS

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

pip3 install scipy matplotlib opencv-python

pip3 install flwr[simulation] flwr-datasets[vision] tensorflow-probability==0.7 keras-mdn-layer

pip3 install keras==2.6.0

pip3 install numpy==1.21.0

tensorboard --logdir=Python/odometry/models/neural_loop_closure_v2
