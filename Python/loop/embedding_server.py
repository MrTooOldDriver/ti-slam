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
from data_tools import get_positive_negative_samples, get_image, load_eval_data
from os.path import join, dirname
import yaml
from random import randrange
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, GetPropertiesIns, GetPropertiesRes, GetParametersIns, \
    GetParametersRes, Status, Code, parameters_to_ndarrays, ndarrays_to_parameters,NDArrays
from flwr.server.strategy import FedAvg
from flwr.server.app import ServerConfig

from embedding_client import EmbeddingClient

# === Load configuration and list of training data ===
with open(join(currentdir, 'config.yaml'), 'r') as f:
    cfg = yaml.safe_load(f)

datatype = cfg['training_opt']['dataset']
if datatype == 'handheld':
    dataroot = cfg['handheld_data']['dataroot']
    loop_path = cfg['handheld_data']['loop_path']
    all_experiments = cfg['handheld_data']['all_exp_files']
    training_experiments = all_experiments[0:cfg['handheld_data']['total_training']]
    n_training = len(training_experiments)
else:
    dataroot = cfg['robot_data']['dataroot']
    loop_path = cfg['robot_data']['loop_path']
    all_experiments = cfg['robot_data']['all_exp_files']
    training_experiments = all_experiments[0:cfg['robot_data']['total_training']]
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
print("Building network model: ", MODEL_NAME)
model_dir = join('./models', MODEL_NAME)

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
    network = base_network(input_size, descriptor_size, trainable=True)
    network_train = build_neural_embedding((img_h, img_w, 1), network, margin_loss)
    optimizer = Adam(lr=lr_rate)
    network_train.compile(loss=None, optimizer=optimizer)
    network_train.summary()
    return network_train

def serverside_eval(parameters):
    model = model_setup()
    model.set_weights(parameters)

    # === Load validation triplets ===
    # Validation files are the same with test file as we dont use it to learn any hyperparameters
    validation_experiments = all_experiments[cfg['robot_data']['total_training']:]
    validation_triplets = load_validation_stack(loop_path, dataroot, validation_experiments, img_h, img_w, img_c, adjacent_frame)
    print('Validation size: ', np.shape(validation_triplets))

    load_eval_data
    
    model.evaluate()    
    
def start_experiment(
    num_rounds = 2,
    client_pool_size = 2,
    num_iterations = None,
    fraction_fit = 1.9,
    min_fit_clients = 2,
    batch_size = 1,
    iid_alpha=1000.0):


    def client_fn(cid: str, client_index):
        return EmbeddingClient(cid, client_index)
    
    
    






