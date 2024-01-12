"""
Training deep Visual-Inertial odometry from pseudo ground truth
"""

import os
os.environ['KERAS_BACKEND']='tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import shutil
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import h5py
# import matplotlib as mpl
# mpl.use('Agg')
import numpy as np
import glob
import os
from os.path import join
import yaml
from utility.networks import build_neural_odometry
from utility.data_tools import odom_validation_stack_hallucination, load_hallucination_data, load_odom_data

import json

import tensorflow.compat.v1.keras.backend as K
from tensorflow.compat.v1.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)


import flwr as fl
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, GetPropertiesIns, GetPropertiesRes, GetParametersIns, \
    GetParametersRes, Status, Code, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common import Metrics
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from typing import Dict, List, Tuple

NUM_CLIENTS = 0
NUM_ROUNDS = 2

def model_setup():
    with open(join(currentdir, 'config.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)

    # Training setting

    base_model_name = cfg['training_opt']['base_model_name']
    is_first_stage = cfg['training_opt']['is_first_stage']

    # Model setting
    MODEL_NAME = cfg['nn_opt']['tio_prob_params']['nn_name']
    n_mixture = cfg['nn_opt']['tio_prob_params']['n_mixture']
    IMU_LENGTH = cfg['nn_opt']['tio_prob_params']['imu_length']

    model_dir = join('./models', MODEL_NAME)
    batch_size = 9

    print("Building network model: ", MODEL_NAME, ", with IMU length", IMU_LENGTH)
    model = build_neural_odometry(cfg['nn_opt']['tio_prob_params'], imu_length=IMU_LENGTH, isfirststage=is_first_stage,
                                         base_model_name=base_model_name, n_mixture=int(n_mixture))
    model.summary(line_length=120)
    return model

class DeeptioClient(fl.client.NumPyClient):

    def __init__(self, cid: str, log_progress: bool = False):
        self.cid = cid
        self.log_progress = log_progress
        self.model = model_setup()
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

    def fit(self, parameters, config):
        
        self.model.set_weights(parameters)

        print('For thermal-IMU ONLY!')

        with open(join(currentdir, 'config.yaml'), 'r') as f:
            cfg = yaml.safe_load(f)
        # Training setting
        data_type = 'turtle' # default is turtle, maybe updated for later use (based on self.index)
        if(data_type=='turtle'):
            data_dir = cfg['training_opt']['data_dir_turtle']
            hallucination_dir = cfg['training_opt']['rgb_feature_dir_turtle']
        else:
            data_dir = cfg['training_opt']['data_dir_handheld']
            hallucination_dir = cfg['training_opt']['rgb_feature_dir_handheld']            

        batch_size = 9
        base_model_name = cfg['training_opt']['base_model_name']
        is_first_stage = cfg['training_opt']['is_first_stage']

        MODEL_NAME = cfg['nn_opt']['tio_prob_params']['nn_name']
        n_mixture = cfg['nn_opt']['tio_prob_params']['n_mixture']
        IMU_LENGTH = cfg['nn_opt']['tio_prob_params']['imu_length']

        model_dir = join('./models', MODEL_NAME)
        # Training with validation set
        checkpoint_path = join('./models', MODEL_NAME, 'best').format('h5')
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path) # was os.remove
        checkpointer = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', mode='min', save_best_only=True,
                                    verbose=1)

        tensor_board = TensorBoard(log_dir=join(model_dir, 'logs'), histogram_freq=0)
        training_loss = []

        # just use a few of it each, not all; the changes are on the index. 
        validation_files = sorted(glob.glob(join(data_dir, 'test', '*.h5')))[self.train_size*self.index:self.train_size*(self.index+1)]
        print(validation_files)
        # just use a few of it each, not all; the changes are on the index.
        hallucination_val_files = sorted(glob.glob(join(hallucination_dir, 'val', '*.h5')))[self.train_size*self.index:self.train_size*(self.index+1)]
        print(join(hallucination_dir, 'val', '*.h5'))
        print(hallucination_val_files)

        x_thermal_val_1, x_thermal_val_2, x_imu_val_t, y_val_t, y_rgb_feat_val_t = odom_validation_stack_hallucination(validation_files,
                                                                                                                    hallucination_val_files,
                                                                                                                    sensor='thermal',
                                                                                                                    imu_length=IMU_LENGTH)
        len_val_i = y_val_t.shape[0]

        print('Final thermal validation shape:', np.shape(x_thermal_val_1), np.shape(y_val_t), np.shape(y_rgb_feat_val_t))

        # grap training files
        all_training = sorted(glob.glob(join(data_dir, 'train', '*.h5')))

        training_files = all_training[self.index*self.train_size:(self.index+1)*self.train_size]
        n_training_files = len(training_files)
        training_file_idx = np.arange(1 + self.index*self.train_size, n_training_files + 1 + self.index*self.train_size)
        seq_len = np.arange(n_training_files)

        for e in range(1): #201
            print("|-----> epoch %d" % e)
            np.random.shuffle(seq_len)
            for i in range(0, n_training_files):

                training_file = data_dir + '/train/' + data_type + '_seq_' + str(training_file_idx[seq_len[i]]) + '.h5'
                hallucination_file = hallucination_dir + '/train/rgb_feat_seq_' + str(training_file_idx[seq_len[i]]) + '.h5'
                print('---> Loading training file: turtle_seq_', str(training_file_idx[seq_len[i]]), '.h5',
                    '---> Loading hallucinatio file: rgb_feat_seq_', str(training_file_idx[seq_len[i]]), '.h5')

                n_chunk, x_thermal_t, x_imu_t, y_t = load_odom_data(training_file, 'thermal')
                n_chunk_feat, y_rgb_feat_t = load_hallucination_data(hallucination_file)

                # generate random length sequences
                len_x_i = x_thermal_t[0].shape[0]  # ex: length of sequence is 300

                range_seq = np.arange(len_x_i - batch_size - 1)
                np.random.shuffle(range_seq)
                for j in range(len(range_seq) // (batch_size - 1)):
                    x_thermal_1, x_thermal_2, x_imu, y_label, y_rgb_feat = [], [], [], [], []
                    starting = range_seq[j * (batch_size - 1)]
                    seq_idx_1 = range(starting, starting + (batch_size - 1))
                    seq_idx_2 = range(starting + 1, starting + batch_size)
                    x_thermal_1.extend(x_thermal_t[0][seq_idx_1, :, :, :])
                    x_thermal_2.extend(x_thermal_t[0][seq_idx_2, :, :, :])

                    x_imu.extend(x_imu_t[0][seq_idx_2, 0:IMU_LENGTH, :])  # for 10 imu data
                    y_label.extend(y_t[0][seq_idx_1, :])
                    y_rgb_feat.extend(y_rgb_feat_t[0][seq_idx_1, :, :])

                    x_thermal_1, x_thermal_2, x_imu, y_label, y_rgb_feat = np.array(x_thermal_1), np.array(x_thermal_2), \
                                                                        np.array(x_imu), np.array(y_label), np.array(
                        y_rgb_feat)

                    # for flownet
                    x_thermal_1 = np.repeat(x_thermal_1, 3, axis=-1)
                    x_thermal_2 = np.repeat(x_thermal_2, 3, axis=-1)

                    y_label = np.expand_dims(y_label, axis=1)

                    print('Training data:', np.shape(x_thermal_1), np.shape(x_thermal_2), np.shape(x_imu))
                    print('Epoch: ', str(e), ', Sequence:', str(i), ', Batch: ', str(j), ', Start at index: ', str(starting))

                    if i == len(seq_len) - 1 and j == (len(range_seq) // (batch_size - 1)) - 1:
                        if int(is_first_stage) == 1:
                            history = self.model.fit(x={'image_1': x_thermal_1, 'image_2': x_thermal_2, 'imu_data': x_imu},
                                                y={'time_distributed_1': y_label[:, :, 0:3], 'time_distributed_2': y_label[:, :, 3:6],
                                                'flatten_rgb': y_rgb_feat},
                                                validation_data=(
                                                    [x_thermal_val_1[0:len_val_i, :, :, :, :],
                                                    x_thermal_val_2[0:len_val_i, :, :, :, :],
                                                    x_imu_val_t[0:len_val_i, :, :]],
                                                    [y_val_t[:, :, 0:3],
                                                    y_val_t[:, :, 3:6], y_rgb_feat_val_t[0:len_val_i, :, :]]),
                                                batch_size=batch_size - 1, shuffle='batch', epochs=1,
                                                callbacks=[checkpointer, tensor_board], verbose=1)
                            training_loss.append(history.history['loss'])
                        else:
                            print('Second stage!')
                            history = self.model.fit({'image_1': x_thermal_1, 'image_2': x_thermal_2, 'imu_data': x_imu},
                                                {'time_distributed_1': y_label[:, :, 0:3], 'time_distributed_2': y_label[:, :, 3:6],
                                                'flatten_rgb': y_rgb_feat},
                                                validation_data=(
                                                    [x_thermal_val_1[0:len_val_i, :, :, :, :],
                                                    x_thermal_val_2[0:len_val_i, :, :, :, :],
                                                    x_imu_val_t[0:len_val_i, :, :]],
                                                    [y_val_t[:, :, 0:3],
                                                    y_val_t[:, :, 3:6], y_rgb_feat_val_t[0:len_val_i, :, :]]),
                                                batch_size=batch_size - 1, shuffle='batch', epochs=1,
                                                callbacks=[checkpointer, tensor_board], verbose=1)
                            training_loss.append(history.history['loss'])

                    else:
                        self.model.fit({'image_1': x_thermal_1, 'image_2': x_thermal_2, 'imu_data': x_imu},
                                {'time_distributed_1': y_label[:, :, 0:3], 'time_distributed_2': y_label[:, :, 3:6],
                                'flatten_rgb': y_rgb_feat},
                                batch_size=batch_size - 1, shuffle='batch', epochs=1, verbose=1)

            if ((e % 25) == 0):
                self.model.save(join(model_dir, self.cid, str(e).format('h5')))

        print("Training for model has finished!")

        print('Saving training loss ....')
        train_loss = np.array(training_loss)
        loss_file_save = join(model_dir, self.cid, 'training_loss.' + MODEL_NAME +'.h5')
        with h5py.File(loss_file_save, 'w') as hf:
            hf.create_dataset('train_loss', data=train_loss)

        print('Saving nn options ....')
        with open(join(model_dir, self.cid, 'nn_opt.json'), 'w') as fp:
            json.dump(cfg['nn_opt']['tio_prob_params'], fp)

        print('Finished training client ', self.cid, str(n_training_files), ' trajectory!') 
        return self.model.get_weights(), self.train_size, {}
    
    def evaluate(self, parameters, config):
        with open(join(currentdir, 'config.yaml'), 'r') as f:
            cfg = yaml.safe_load(f)
        # Training setting
        data_type = 'turtle' # default is turtle, maybe updated for later use (based on self.index)
        if(data_type=='turtle'):
            data_dir = cfg['training_opt']['data_dir_turtle']
            hallucination_dir = cfg['training_opt']['rgb_feature_dir_turtle']
        else:
            data_dir = cfg['training_opt']['data_dir_handheld']
            hallucination_dir = cfg['training_opt']['rgb_feature_dir_handheld']    
        MODEL_NAME = cfg['nn_opt']['tio_prob_params']['nn_name']
        n_mixture = cfg['nn_opt']['tio_prob_params']['n_mixture']
        IMU_LENGTH = cfg['nn_opt']['tio_prob_params']['imu_length']
        # just use a few of it each, not all; the changes are on the index. 
        validation_files = sorted(glob.glob(join(data_dir, 'test', '*.h5')))[self.train_size*self.index:self.train_size*(self.index+1)]
        print(validation_files)
        # just use a few of it each, not all; the changes are on the index.
        hallucination_val_files = sorted(glob.glob(join(hallucination_dir, 'val', '*.h5')))[self.train_size*self.index:self.train_size*(self.index+1)]
        print(join(hallucination_dir, 'val', '*.h5'))
        print(hallucination_val_files)

        x_thermal_val_1, x_thermal_val_2, x_imu_val_t, y_val_t, y_rgb_feat_val_t = odom_validation_stack_hallucination(validation_files,
                                                                                                                    hallucination_val_files,
                                                                                                                    sensor='thermal',
                                                                                                                    imu_length=IMU_LENGTH)
        len_val_i = y_val_t.shape[0]        
        
        self.model.set_weights(parameters)
        loss = self.model.evaluate(x=[x_thermal_val_1[0:len_val_i, :, :, :, :], x_thermal_val_2[0:len_val_i, :, :, :, :],x_imu_val_t[0:len_val_i, :, :]],
                                    y=[y_val_t[:, :, 0:3],y_val_t[:, :, 3:6], y_rgb_feat_val_t[0:len_val_i, :, :]])

        return loss, {"loss": loss}       

def get_client_fn():
    def client_fn(cid:str):
        return DeeptioClient(cid)
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
        # Training setting
        data_type = 'turtle' # default is turtle, maybe updated for later use (based on self.index)
        if(data_type=='turtle'):
            data_dir = cfg['training_opt']['data_dir_turtle']
            hallucination_dir = cfg['training_opt']['rgb_feature_dir_turtle']
        else:
            data_dir = cfg['training_opt']['data_dir_handheld']
            hallucination_dir = cfg['training_opt']['rgb_feature_dir_handheld']    
        MODEL_NAME = cfg['nn_opt']['tio_prob_params']['nn_name']
        n_mixture = cfg['nn_opt']['tio_prob_params']['n_mixture']
        IMU_LENGTH = cfg['nn_opt']['tio_prob_params']['imu_length']
        # just use a few of it each, not all; the changes are on the index. 
        validation_files = sorted(glob.glob(join(data_dir, 'test', '*.h5')))[:1]
        print(validation_files)
        # just use a few of it each, not all; the changes are on the index.
        hallucination_val_files = sorted(glob.glob(join(hallucination_dir, 'val', '*.h5')))[:1]
        print(join(hallucination_dir, 'val', '*.h5'))
        print(hallucination_val_files)

        x_thermal_val_1, x_thermal_val_2, x_imu_val_t, y_val_t, y_rgb_feat_val_t = odom_validation_stack_hallucination(validation_files,
                                                                                                                    hallucination_val_files,
                                                                                                                    sensor='thermal',
                                                                                                                    imu_length=IMU_LENGTH)
        len_val_i = y_val_t.shape[0]        
        
        with tf.device('/cpu:0'):
            model = model_setup()             
            model.set_weights(parameters)
            loss = model.evaluate(x=[x_thermal_val_1[0:len_val_i, :, :, :, :], x_thermal_val_2[0:len_val_i, :, :, :, :],x_imu_val_t[0:len_val_i, :, :]],
                                  y=[y_val_t[:, :, 0:3],y_val_t[:, :, 3:6], y_rgb_feat_val_t[0:len_val_i, :, :]])
        print("server round "+ str(server_round))
        if(server_round % 5 == 4):
            model.save(join("server_deeptio_model", str(server_round).format('h5'))) 
               
        return loss, {"loss": loss}
        # return 0.7, {"loss": 0.7} # for debugging cases           
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
        #actor_kwargs={
        #    "on_actor_init_fn": enable_tf_gpu_growth  # Enable GPU growth upon actor init
            # does nothing if `num_gpus` in client_resources is 0.0
        #},
    )        


if __name__ == "__main__":
    # enable_tf_gpu_growth()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    main()