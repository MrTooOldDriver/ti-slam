import gc
import math
import os
os.environ['KERAS_BACKEND']='tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import numpy as np
np.random.seed(0)
from pylab import *

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)
tf_config=tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth=True
import tensorflow.compat.v1.keras.backend as K
from tensorflow.compat.v1.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler

from utility.networks import build_neural_loop_closure
from utility.data_tools import get_pose_pairs, get_image
import os
from os.path import join
import shutil
import yaml
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, GetPropertiesIns, GetPropertiesRes, GetParametersIns, \
    GetParametersRes, Status, Code, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common import Metrics
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from typing import Dict, List, Tuple
import random

NUM_CLIENTS = 0
NUM_ROUNDS = 2
BEST_LOSS = 1000

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(0)
    tf.experimental.numpy.random.seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def load_validation_stack(loop_path, dataroot, validation_exps, img_h, img_w, img_c):
    # Reserve the validation stack data
    total_val_length = 0
    for i, validation_exp in enumerate(validation_exps):
        pose_data = get_pose_pairs(loop_path, validation_exp)
        total_val_length += len(pose_data)
    # x_image_1 = np.zeros((total_val_length, 1, img_h, img_w, 3), dtype=np.float16)
    # x_image_2 = np.zeros((total_val_length, 1, img_h, img_w, 3), dtype=np.float16)
    x_image_1 = np.zeros((total_val_length, 1, img_h, img_w, 3), dtype=np.float32)
    x_image_2 = np.zeros((total_val_length, 1, img_h, img_w, 3), dtype=np.float32)
    y_pose = np.zeros((total_val_length, 1, 6))
    print('Allocated validation data: ' + str(np.shape(x_image_1)) + ' - ' + str(np.shape(x_image_2)) + ' - ' + str(np.shape(y_pose)))

    # Loop for all experimental folders
    val_idx = 0
    for i, validation_exp in enumerate(validation_exps):
        pose_data = get_pose_pairs(loop_path, validation_exp)

        img_root_path = dataroot + '/' + validation_exp + '/thermal/'
        for j in range(len(pose_data)):
            image_1_path = img_root_path + pose_data[j].split(',')[1]
            image_2_path = img_root_path + pose_data[j].split(',')[3]

            img_1 = get_image(image_1_path)
            img_1 = np.repeat(img_1, 3, axis=-1)
            img_2 = get_image(image_2_path)
            img_2 = np.repeat(img_2, 3, axis=-1)

            x_image_1[val_idx, 0, :, :, :] = img_1
            x_image_2[val_idx, 0, :, :, :] = img_2
            for idx_p in range(6):
                y_pose[val_idx, 0, idx_p] = float(pose_data[j].split(',')[4+idx_p])
            val_idx += 1

    return x_image_1, x_image_2, y_pose

def model_setup():
    with open(join(currentdir, 'config.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)    
    MODEL_NAME = cfg['nn_opt']['loop_params']['nn_name']        
    print("Building network model: ", MODEL_NAME)
    model_dir = join('./models', MODEL_NAME)
    # === Model Definition ===
    network_train = build_neural_loop_closure(cfg['nn_opt']['loop_params'])
    return network_train


class LoopPoseClient(fl.client.NumPyClient):

    def __init__(self, cid: str, log_progress: bool = False):
        self.cid = cid
        self.log_progress = log_progress
        self.model = model_setup()
        self.train_size = 3
        self.val_size = 1
        self.index = int(cid)
        set_seed(0)
        print('cid:', self.cid, 'index:', self.index)
   
    def get_parameters(self, config):
        print("client "+ self.cid + " giving parameters to server")
        return self.model.get_weights()

    def set_parameters(self, parameters,config):
        print("client "+ self.cid + " received parameters from server")
        return self.model.set_weights(parameters)

    def fit(self, parameters, config):
        
        self.model.set_weights(parameters)

        # === Load configuration and list of training data ===
        with open(join(currentdir, 'config.yaml'), 'r') as f:
            cfg = yaml.safe_load(f)

        datatype = cfg['train_loop_pose_opt']['dataset']

        if datatype == 'handheld':
            dataroot = cfg['loop_handheld_data']['dataroot']
            loop_path = cfg['loop_handheld_data']['loop_path']
            all_experiments = cfg['loop_handheld_data']['all_exp_files']
            if self.index >= (cfg['loop_robot_data']['total_training'] / self.train_size):
                print("client %s outbound, randmly select  %s from training data" % (self.index, self.train_size))
                np.random.seed(0)
                training_experiments = np.random.choice(all_experiments[0:cfg['loop_robot_data']['total_training']], self.train_size, replace=False)
            else:
                print("client %s select %s from training data" % (self.index, self.train_size))
                training_experiments = all_experiments[self.index*self.train_size:(self.index+1)*self.train_size]
            n_training = len(training_experiments)
        else:
            dataroot = cfg['loop_robot_data']['dataroot']
            loop_path = cfg['loop_robot_data']['loop_path']
            all_experiments = cfg['loop_robot_data']['all_exp_files']
            if self.index >= (cfg['loop_robot_data']['total_training'] / self.train_size):
                print("client %s outbound, randmly select  %s from training data" % (self.index, self.train_size))
                np.random.seed(0)
                training_experiments = np.random.choice(all_experiments[0:cfg['loop_robot_data']['total_training']], self.train_size, replace=False)
            else:
                print("client %s select %s from training data" % (self.index, self.train_size))
                training_experiments = all_experiments[self.index*self.train_size:(self.index+1)*self.train_size]
            n_training = len(training_experiments)
        
        MODEL_NAME = cfg['nn_opt']['loop_params']['nn_name']
        img_h = cfg['nn_opt']['loop_params']['img_h']
        img_w = cfg['nn_opt']['loop_params']['img_w']
        img_c = cfg['nn_opt']['loop_params']['img_c']
        n_epoch = cfg['nn_opt']['loop_params']['epoch']
        batch_size = cfg['nn_opt']['loop_params']['batch_size']
        input_size = (img_h, img_w, img_c)
                    
        model_dir = join('./models', MODEL_NAME)

        checkpoint_path = join('./models', MODEL_NAME, 'best').format('h5')
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
        # regulate learning rate
        def step_decay(epoch):
            initial_lrate = cfg['nn_opt']['loop_params']['lr_rate']  # 0.001, 0.0001
            drop = 0.75
            epochs_drop = 25.0
            lrate = initial_lrate * math.pow(drop,
                                            math.floor((1 + epoch) / epochs_drop))
            print('Learning rate: ' + str(lrate))
            return lrate
        lrate = LearningRateScheduler(step_decay)
        # === Load validation poses ===
        # Validation files are the same with test file as we dont use it to learn any hyperparameters
        val_starting = cfg['loop_robot_data']['total_training']
        if self.index >= (cfg['loop_robot_data']['total_training'] / self.train_size):
            print("client %s outbound, randmly select  %s from validation data" % (self.index, self.val_size))
            np.random.seed(0)
            validation_experiments = np.random.choice(all_experiments[val_starting:], self.val_size, replace=False)
        else:
            print("client %s select %s from validation data" % (self.index, self.val_size))
            validation_experiments = all_experiments[val_starting+self.index*self.val_size:val_starting+(self.index+1)*self.val_size] # dir/file names for validation

        x_val_img_1, x_val_img_2, y_val = load_validation_stack(loop_path, dataroot, validation_experiments, img_h, img_w, img_c)
        print('Validation size: ' + str(np.shape(x_val_img_1)) + ' - ' + str(np.shape(x_val_img_2)))    

        # === Training loops ===
        for e in range(0, 5): # n_epoch, just change it small for testing the code
            print("|-----> epoch %d" % e)
            # Shuffle training sequences
            np.random.shuffle(training_experiments)
            for i in range(0, n_training): # training experiments/folders
                # Load all positive-negative examples in particular sequence
                pose_data = get_pose_pairs(loop_path, training_experiments[i])
                data_length = len(pose_data)

                # Important! Shuffle the data!
                np.random.shuffle(pose_data)
                print('Epoch: ', str(e), ', Sequence: ', str(i), ' - ', training_experiments[i])

                batch_iteration = int(data_length / batch_size) # For all positive pairs
                for j in range(0, batch_iteration): # how many batch per sequences/exp
                    # Initialize training batches
                    # x_img_1 = np.zeros((batch_size, 1, img_h, img_w, 3), dtype=np.float16)
                    # x_img_2 = np.zeros((batch_size, 1, img_h, img_w, 3), dtype=np.float16)
                    x_img_1 = np.zeros((batch_size, 1, img_h, img_w, 3), dtype=np.float32)
                    x_img_2 = np.zeros((batch_size, 1, img_h, img_w, 3), dtype=np.float32)
                    y_pose = np.zeros((batch_size, 1, 6))
                    # Get the image batch
                    for k in range(0, batch_size):
                        img_root_path = dataroot + '/' + training_experiments[i] + '/thermal/'
                        img_1_path = img_root_path + pose_data[(j*batch_size)+k].split(',')[1]
                        img_2_path = img_root_path + pose_data[(j*batch_size)+k].split(',')[3]

                        img_1 = get_image(img_1_path)
                        img_1 = np.repeat(img_1, 3, axis=-1)

                        img_2 = get_image(img_2_path)
                        img_2 = np.repeat(img_2, 3, axis=-1)

                        x_img_1[k, 0, :, :, :] = img_1
                        x_img_2[k, 0, :, :, :] = img_2

                        for idx_p in range(6):
                            y_pose[k, 0, idx_p] = float(pose_data[(j*batch_size)+k].split(',')[4 + idx_p])

                    # Implement Get batch hard for hard triplet loss here!!!
                    if i == (n_training - 1) and j == (batch_iteration - 1):
                        # Train on batch and validate
                        self.model.fit(x=[x_img_1, x_img_2], y=[y_pose[:, :, 0:3], y_pose[:, :, 3:6]], verbose=1,
                                        validation_data=([x_val_img_1, x_val_img_2],
                                                        [y_val[:, :, 0:3], y_val[:, :, 3:6]]),
                                        # callbacks=[checkpointer, lrate, tensor_board])
                                        callbacks = [])
                    else:
                        self.model.fit(x=[x_img_1, x_img_2], y=[y_pose[:, :, 0:3], y_pose[:, :, 3:6]], verbose=1) # Train on batch
        return self.model.get_weights(), self.train_size, {}

    def evaluate(self, parameters, config):
        # === Load configuration and list of training data ===
        with open(join(currentdir, 'config.yaml'), 'r') as f:
            cfg = yaml.safe_load(f)        
        
        datatype = cfg['train_loop_pose_opt']['dataset']
        if datatype == 'handheld':
            dataroot = cfg['loop_handheld_data']['dataroot']
            loop_path = cfg['loop_handheld_data']['loop_path']
            all_experiments = cfg['loop_handheld_data']['all_exp_files']
        else:
            dataroot = cfg['loop_robot_data']['dataroot']
            loop_path = cfg['loop_robot_data']['loop_path']
            all_experiments = cfg['loop_robot_data']['all_exp_files']
        
        MODEL_NAME = cfg['nn_opt']['loop_params']['nn_name']
        img_h = cfg['nn_opt']['loop_params']['img_h']
        img_w = cfg['nn_opt']['loop_params']['img_w']
        img_c = cfg['nn_opt']['loop_params']['img_c']        

        # === Load validation poses ===
        # Validation files are the same with test file as we dont use it to learn any hyperparameters
        val_starting = cfg['loop_robot_data']['total_training']
        if self.index >= (cfg['loop_robot_data']['total_training'] / self.train_size):
            print("client %s outbound, randmly select  %s from validation data" % (self.index, self.val_size))
            np.random.seed(0)
            validation_experiments = np.random.choice(all_experiments[val_starting:], self.val_size, replace=False)
        else:
            print("client %s select %s from validation data" % (self.index, self.val_size))
            validation_experiments = all_experiments[val_starting+self.index*self.val_size:val_starting+(self.index+1)*self.val_size] # dir/file names for validation

        x_val_img_1, x_val_img_2, y_val = load_validation_stack(loop_path, dataroot, validation_experiments, img_h, img_w, img_c)
        print('Validation size: ' + str(np.shape(x_val_img_1)) + ' - ' + str(np.shape(x_val_img_2)))    

        self.model.set_weights(parameters)
        loss = self.model.evaluate(x=[x_val_img_1, x_val_img_2], y=[y_val[:, :, 0:3], y_val[:, :, 3:6]])
        del self.model
        self.model = None
        tf.compat.v1.reset_default_graph()
        gc.collect()
        return loss[0], self.val_size, {"loss": loss[0]}

def get_client_fn():
    def client_fn(cid:str):
        return LoopPoseClient(cid)
    return client_fn

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"loss": sum(losses) / sum(examples)}   

def get_evaluate_fn():
    def evaluate(server_round, parameters, config):
        with tf.device('/gpu:3'):
            # === Load configuration and list of training data ===
            with open(join(currentdir, 'config.yaml'), 'r') as f:
                cfg = yaml.safe_load(f)        
            
            datatype = cfg['train_loop_pose_opt']['dataset']
            if datatype == 'handheld':
                dataroot = cfg['loop_handheld_data']['dataroot']
                loop_path = cfg['loop_handheld_data']['loop_path']
                all_experiments = cfg['loop_handheld_data']['all_exp_files']
            else:
                dataroot = cfg['loop_robot_data']['dataroot']
                loop_path = cfg['loop_robot_data']['loop_path']
                all_experiments = cfg['loop_robot_data']['all_exp_files']
            
            MODEL_NAME = cfg['nn_opt']['loop_params']['nn_name']
            img_h = cfg['nn_opt']['loop_params']['img_h']
            img_w = cfg['nn_opt']['loop_params']['img_w']
            img_c = cfg['nn_opt']['loop_params']['img_c']        

            # === Load validation poses ===
            # Validation files are the same with test file as we dont use it to learn any hyperparameters
            val_starting = cfg['loop_robot_data']['total_training']
            validation_experiments = all_experiments[val_starting:]

            x_val_img_1, x_val_img_2, y_val = load_validation_stack(loop_path, dataroot, validation_experiments, img_h, img_w, img_c)
            print('Validation size: ' + str(np.shape(x_val_img_1)) + ' - ' + str(np.shape(x_val_img_2)))   

        
            model = model_setup()             
            model.set_weights(parameters)
            loss = model.evaluate(x=[x_val_img_1, x_val_img_2], y=[y_val[:, :, 0:3], y_val[:, :, 3:6]])
        print("server round " + str(server_round))
        global BEST_LOSS
        if loss[0] < BEST_LOSS:
            BEST_LOSS = loss[0]
            model.save(join("server_model", "best").format('h5'))
            print("best model saved loss: " + str(BEST_LOSS))
        print("BEST_LOSS: " + str(BEST_LOSS))
        del model
        model = None
        tf.compat.v1.reset_default_graph()
        gc.collect()
        return loss, {"loss": loss}
    return evaluate


def main():
    num_clients = 3
    # strategy = fl.server.strategy.FedAvg(
    #     fraction_fit=1,  #
    #     fraction_evaluate=1,  # 
    #     min_fit_clients=1,  #
    #     min_evaluate_clients=num_clients,  # 
    #     min_available_clients=int(
    #         num_clients * 1
    #     ),  
    #     evaluate_metrics_aggregation_fn=weighted_average,  # aggregates federated metrics
    #     evaluate_fn=get_evaluate_fn(),  # global evaluation function
    # )
    strategy = fl.server.strategy.FedTrimmedAvg(
        fraction_fit=1,  #
        fraction_evaluate=1,  # 
        min_fit_clients=1,  #
        min_evaluate_clients=num_clients,  # 
        min_available_clients=int(
            num_clients * 1
        ),  
        evaluate_metrics_aggregation_fn=weighted_average,  # aggregates federated metrics
        evaluate_fn=get_evaluate_fn(),  # global evaluation function
    )
    client_resources = {
        "num_gpus": 1,
        "num_cpus": 8
    }
    ray_init_args = {
        "num_cpus": 56,
        "num_gpus": 3
    }
    fl.simulation.start_simulation(
        client_fn=get_client_fn(),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args = ray_init_args,
        actor_kwargs={
           "on_actor_init_fn": enable_tf_gpu_growth  # Enable GPU growth upon actor init
        },
    )   


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))  
    set_seed(0)
    main()