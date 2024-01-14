import os
os.environ['KERAS_BACKEND']='tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import tensorflow as tf
tf_config=tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=tf_config)

import numpy as np
np.random.seed(0)

from pylab import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from networks import base_network, build_neural_embedding
from data_tools import get_positive_negative_samples, get_image
from os.path import join, dirname
import yaml
from random import randrange
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, GetPropertiesIns, GetPropertiesRes, GetParametersIns, \
    GetParametersRes, Status, Code, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common import Metrics
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from typing import Dict, List, Tuple

NUM_CLIENTS = 2
NUM_ROUNDS = 2
def load_validation_stack(loop_path, dataroot, validation_exps, img_h, img_w, img_c, adjacent_frame):
    # Reserve the validation stack data
    total_val_length = 0
    for i, validation_exp in enumerate(validation_exps):
        pos_data, neg_data = get_positive_negative_samples(loop_path, validation_exp)
        total_val_length += len(pos_data)
    val_triplets = [np.zeros((total_val_length, img_h, img_w, 1)) for a in range(3)]

    # Loop for all experimental folders
    triplet_idx = 0
    for i, validation_exp in enumerate(validation_exps):
        pos_data, neg_data = get_positive_negative_samples(loop_path, validation_exp)
        np.random.shuffle(neg_data)
        img_root_path = dataroot + '/' + validation_exp + '/thermal/'
        for j in range(len(pos_data)):
            anchor_path = img_root_path + pos_data[j].split(',')[1]
            pos_path = img_root_path + pos_data[j].split(',')[3]

            anchor_idx = int(pos_data[j].split(',')[0])
            range_not_allowed = range(max(0, anchor_idx - adjacent_frame),
                                      anchor_idx + adjacent_frame)  # list of not allowed negative index

            # Get random negative example, but not within the adjacent frames of anchor images
            for trial in range(100):
                rand_ned_idx = randrange(len(neg_data))  # gen random idx whithin the range of negative samples

                neg_idx = int(neg_data[rand_ned_idx].split(',')[0])  # actual index of negative image
                if not neg_idx is range_not_allowed:  # check if the image idx is not within the adjacent anchor frames
                    neg_path = img_root_path + neg_data[rand_ned_idx].split(',')[1]
                    break

            anchor_img = get_image(anchor_path)
            # anchor_img = np.repeat(anchor_img, 3, axis=-1)
            pos_img = get_image(pos_path)
            # pos_img = np.repeat(pos_img, 3, axis=-1)
            neg_img = get_image(neg_path)
            # neg_img = np.repeat(neg_img, 3, axis=-1)

            val_triplets[0][triplet_idx, :, :, :] = anchor_img
            val_triplets[1][triplet_idx, :, :, :] = pos_img
            val_triplets[2][triplet_idx, :, :, :] = neg_img
            triplet_idx += 1

    return val_triplets

def model_setup():
    """
    Setup the neural loop model
    """
    with open(join(currentdir, 'config.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)
    img_h = cfg['training_opt']['thermal_params']['img_h']
    img_w = cfg['training_opt']['thermal_params']['img_w']
    img_c = cfg['training_opt']['thermal_params']['img_c']    
    
    descriptor_size = cfg['training_opt']['thermal_params']['descriptor_size']
    input_size = (img_h, img_w, img_c)
    margin_loss = cfg['training_opt']['thermal_params']['margin_loss']
    lr_rate = cfg['training_opt']['thermal_params']['lr_rate']    
    network = base_network(input_size, descriptor_size, trainable=True)
    network_train = build_neural_embedding((img_h, img_w, 1), network, margin_loss)
    optimizer = Adam(lr=lr_rate)
    network_train.compile(loss=None, optimizer=optimizer)
    network_train.summary()
    return network_train, network

class EmbeddingClient(fl.client.NumPyClient):

    def __init__(self, cid: str, log_progress: bool = False):
        self.cid = cid
        self.log_progress = log_progress
        self.model, self.model_saved = model_setup()
        self.train_size = 2
        self.val_size = 1
        global NUM_CLIENTS
        self.index = NUM_CLIENTS
        NUM_CLIENTS = NUM_CLIENTS + 1

    def get_parameters(self, config):
        print("client "+ self.cid + " giving parameters to server")
        return self.model.get_weights()
    
    def set_parameters(self, parameters,config):
        print("client "+ self.cid + " received parameters from server")
        return self.model.set_weights(parameters)
    
    def fit(self, parameters,config):
        self.set_parameters(parameters, config)
        with open(join(currentdir, 'config.yaml'), 'r') as f:
            cfg = yaml.safe_load(f)

        datatype = cfg['training_opt']['dataset'] # handheld or turtle (the robot)
        if datatype == 'handheld':
            dataroot = cfg['handheld_data']['dataroot']
            loop_path = cfg['handheld_data']['loop_path']
            all_experiments = cfg['handheld_data']['all_exp_files']
            training_experiments = all_experiments[self.index*self.train_size:(self.index+1)*self.train_size]
            n_training = len(training_experiments)
        else:
            dataroot = cfg['robot_data']['dataroot']
            loop_path = cfg['robot_data']['loop_path']
            all_experiments = cfg['robot_data']['all_exp_files']
            training_experiments = all_experiments[self.index*self.train_size:(self.index+1)*self.train_size]
            n_training = len(training_experiments)

        MODEL_NAME = cfg['training_opt']['thermal_params']['nn_name']
        lr_rate = cfg['training_opt']['thermal_params']['lr_rate']
        decay = cfg['training_opt']['thermal_params']['decay']
        margin_loss = cfg['training_opt']['thermal_params']['margin_loss']
        img_h = cfg['training_opt']['thermal_params']['img_h']
        img_w = cfg['training_opt']['thermal_params']['img_w']
        img_c = cfg['training_opt']['thermal_params']['img_c']
        adjacent_frame = cfg['training_opt']['thermal_params']['adjacent_frame']
        descriptor_size = cfg['training_opt']['thermal_params']['descriptor_size']
        n_epoch = cfg['training_opt']['thermal_params']['epoch']
        batch_size = cfg['training_opt']['thermal_params']['batch_size']
        input_size = (img_h, img_w, img_c)
        print("Building network model: ", MODEL_NAME, " for client cid", self.cid)
        model_dir = join('./models', self.cid, MODEL_NAME) 

        # --
        checkpoint_path = join('./models', self.cid, MODEL_NAME, 'best').format('h5')
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        checkpointer = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', mode='min', save_best_only=True,
                                    verbose=1)
        tensor_board = TensorBoard(log_dir=join(model_dir, self.cid, 'logs'))

        # === Load validation triplets ===
        # Validation files are the same with test file as we dont use it to learn any hyperparameters
        val_starting = cfg['robot_data']['total_training']
        validation_experiments = all_experiments[val_starting+self.index*self.val_size:val_starting+(self.index+1)*self.val_size] # dir/file names for validation
        validation_triplets = load_validation_stack(loop_path, dataroot, validation_experiments, img_h, img_w, img_c, adjacent_frame)
        print('Validatio size: ', np.shape(validation_triplets))

        # === Training loops ===
        for e in range(0, n_epoch): # epoch
            print("|-----> epoch %d" % e)
            # Shuffle training sequences
            np.random.shuffle(training_experiments)
            for i in range(0, n_training): # training experiments/folders
                # Load all positive-negative examples in particular sequence
                pos_data, neg_data = get_positive_negative_samples(loop_path, training_experiments[i])
                pos_length = len(pos_data)

                # Important! Shuffle both positive and negative pairs!
                np.random.shuffle(pos_data)
                np.random.shuffle(neg_data)
                print('Epoch: ', str(e), ', Sequence: ', str(i), ' - ', training_experiments[i])

                batch_iteration = int(pos_length / batch_size) # For all positive pairs
                for j in range(0, batch_iteration): # how many batch per sequences/exp
                    # Initialize triplets
                    triplets = [np.zeros((batch_size, img_h, img_w, 1)) for a in range(3)]
                    # Get the image batch
                    for k in range(0, batch_size):
                        img_root_path = dataroot + '/' + training_experiments[i] + '/thermal/'
                        anchor_path = img_root_path + pos_data[(j*batch_size)+k].split(',')[1]
                        pos_path = img_root_path + pos_data[(j*batch_size)+k].split(',')[3]

                        anchor_idx = int(pos_data[(j*batch_size)+k].split(',')[0])
                        range_not_allowed = range(max(0, anchor_idx - adjacent_frame),
                                                anchor_idx + adjacent_frame)  # list of not allowed negative index

                        # Get random negative example, but not within the adjacent frames of anchor images
                        for trial in range(100):
                            rand_ned_idx = randrange(len(neg_data))  # gen random idx whithin the range of negative samples

                            neg_idx = int(neg_data[rand_ned_idx].split(',')[0])  # actual index of negative image
                            if not neg_idx is range_not_allowed:  # check if the image idx is not within the adjacent anchor frames
                                neg_path = img_root_path + neg_data[rand_ned_idx].split(',')[1]
                                break

                        anchor_img = get_image(anchor_path)
                        # anchor_img = np.repeat(anchor_img, 3, axis=-1)
                        pos_img = get_image(pos_path)
                        # pos_img = np.repeat(pos_img, 3, axis=-1)
                        neg_img = get_image(neg_path)
                        # neg_img = np.repeat(neg_img, 3, axis=-1)

                        triplets[0][k, :, :, :] = anchor_img
                        triplets[1][k, :, :, :] = pos_img
                        triplets[2][k, :, :, :] = neg_img

                    print(np.shape(triplets))
                    # Implement Get batch hard for hard triplet loss here!!!

                    if i == (n_training - 1) and j == (batch_iteration - 1):
                        # Train on batch and validate
                        self.model.fit(x=[triplets[0], triplets[1], triplets[2]], y=None, verbose=1,
                                        validation_data=([validation_triplets[0], validation_triplets[1], validation_triplets[2]], None),
                                        callbacks=[checkpointer, tensor_board])
                    else:
                        self.model.fit(x=[triplets[0], triplets[1], triplets[2]], y=None, verbose=1) # Train on batch

            if ((e % 10) == 0):
                self.model_saved.save(join(model_dir, self.cid, str(e).format('h5')))
        return self.model.get_weights(), self.train_size, {}
    
    def evaluate(self, parameters, config):
        with open(join(currentdir, 'config.yaml'), 'r') as f:
            cfg = yaml.safe_load(f)

        datatype = cfg['training_opt']['dataset'] # handheld or turtle (the robot)
        if datatype == 'handheld':
            dataroot = cfg['handheld_data']['dataroot']
            loop_path = cfg['handheld_data']['loop_path']
            all_experiments = cfg['handheld_data']['all_exp_files']
        else:
            dataroot = cfg['robot_data']['dataroot']
            loop_path = cfg['robot_data']['loop_path']
            all_experiments = cfg['robot_data']['all_exp_files']

        MODEL_NAME = cfg['training_opt']['thermal_params']['nn_name']
        lr_rate = cfg['training_opt']['thermal_params']['lr_rate']
        decay = cfg['training_opt']['thermal_params']['decay']
        margin_loss = cfg['training_opt']['thermal_params']['margin_loss']
        img_h = cfg['training_opt']['thermal_params']['img_h']
        img_w = cfg['training_opt']['thermal_params']['img_w']
        img_c = cfg['training_opt']['thermal_params']['img_c']
        adjacent_frame = cfg['training_opt']['thermal_params']['adjacent_frame']
        descriptor_size = cfg['training_opt']['thermal_params']['descriptor_size']
        n_epoch = cfg['training_opt']['thermal_params']['epoch']
        batch_size = cfg['training_opt']['thermal_params']['batch_size']
        input_size = (img_h, img_w, img_c)
        # === Load validation triplets ===
        # Validation files are the same with test file as we dont use it to learn any hyperparameters
        validation_experiments = all_experiments[cfg['robot_data']['total_training']+self.index:cfg['robot_data']['total_training']+self.index+self.val_size]
        #validation_experiments = all_experiments[val_starting+self.index*self.val_size:val_starting+(self.index+1)*self.val_size] # dir/file names for validation
        validation_triplets = load_validation_stack(loop_path, dataroot, validation_experiments, img_h, img_w, img_c, adjacent_frame)
        print('Validation size: ', np.shape(validation_triplets))                
        self.model.set_weights(parameters)
        loss = self.model.evaluate(x=[validation_triplets[0], validation_triplets[1], validation_triplets[2]], y=None)
        return loss, {"loss": loss}
    

def get_client_fn():
    def client_fn(cid:str):
        return EmbeddingClient(cid)
    return client_fn

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"loss": sum(losses) / sum(examples)}    

def get_evaluate_fn():
    def evaluate(server_round, parameters, config):
        with open(join(currentdir, 'config.yaml'), 'r') as f:
            cfg = yaml.safe_load(f)

        datatype = cfg['training_opt']['dataset'] # handheld or turtle (the robot)
        if datatype == 'handheld':
            dataroot = cfg['handheld_data']['dataroot']
            loop_path = cfg['handheld_data']['loop_path']
            all_experiments = cfg['handheld_data']['all_exp_files']
        else:
            dataroot = cfg['robot_data']['dataroot']
            loop_path = cfg['robot_data']['loop_path']
            all_experiments = cfg['robot_data']['all_exp_files']

        MODEL_NAME = cfg['training_opt']['thermal_params']['nn_name']
        lr_rate = cfg['training_opt']['thermal_params']['lr_rate']
        decay = cfg['training_opt']['thermal_params']['decay']
        margin_loss = cfg['training_opt']['thermal_params']['margin_loss']
        img_h = cfg['training_opt']['thermal_params']['img_h']
        img_w = cfg['training_opt']['thermal_params']['img_w']
        img_c = cfg['training_opt']['thermal_params']['img_c']
        adjacent_frame = cfg['training_opt']['thermal_params']['adjacent_frame']
        descriptor_size = cfg['training_opt']['thermal_params']['descriptor_size']
        n_epoch = cfg['training_opt']['thermal_params']['epoch']
        batch_size = cfg['training_opt']['thermal_params']['batch_size']
        input_size = (img_h, img_w, img_c)
        # === Load validation triplets ===
        # Validation files are the same with test file as we dont use it to learn any hyperparameters
        validation_experiments = all_experiments[cfg['robot_data']['total_training']:]
        validation_triplets = load_validation_stack(loop_path, dataroot, validation_experiments, img_h, img_w, img_c, adjacent_frame)
        print('Validation size: ', np.shape(validation_triplets))

        with tf.device('/cpu:0'):
            model, model_saved = model_setup()
            model.set_weights(parameters)
            loss = model.evaluate(x=[validation_triplets[0], validation_triplets[1], validation_triplets[2]], y=None)
        print("server round "+ str(server_round))
        if(server_round % 5 == 4):
            model_saved.save(join("server_model", str(server_round).format('h5')))
        return loss, {"loss": loss} 
    return evaluate
       
def main():
    num_clients = 1
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,  #
        fraction_evaluate=1,  # 
        min_fit_clients=1,  #
        min_evaluate_clients=2,  # 
        min_available_clients=int(
            num_clients * 1
        ),  
        evaluate_metrics_aggregation_fn=weighted_average,  # aggregates federated metrics
        evaluate_fn=get_evaluate_fn(),  # global evaluation function
    )
    client_resources = {
        "num_gpus": 1,
        #"num_cpus": 4
    }
    fl.simulation.start_simulation(
        client_fn=get_client_fn(),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy,
        client_resources=client_resources,
        actor_kwargs={
            "on_actor_init_fn": enable_tf_gpu_growth  # Enable GPU growth upon actor init
            # does nothing if `num_gpus` in client_resources is 0.0
        },
    )

if __name__ == "__main__":
    # Enable GPU growth in your main process
    enable_tf_gpu_growth()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    main()

