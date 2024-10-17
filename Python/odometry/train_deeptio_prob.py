"""
Training deep Visual-Inertial odometry from pseudo ground truth
"""

import os

from time_sqe_corresponding import corresponding
os.environ['KERAS_BACKEND']='tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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
import random

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

def main():
    print('For thermal-IMU ONLY!')

    with open(join(currentdir, 'config.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)

    # Training setting
    data_type = cfg['training_opt']['dataset']
    data_dir_key = 'data_dir_' + data_type
    data_dir = cfg['training_opt'][data_dir_key]
    hallucination_dir_key = 'rgb_feature_dir_' + data_type
    hallucination_dir = cfg['training_opt'][hallucination_dir_key]
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

    # Training with validation set
    checkpoint_path = join('./models', MODEL_NAME, 'best').format('h5')
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', mode='min', save_best_only=True,
                                   verbose=1)

    tensor_board = TensorBoard(log_dir=join(model_dir, 'logs'), histogram_freq=0)
    training_loss = []

    if data_type == 'turtle':
        all_exp_files = cfg['loop_robot_data']['all_exp_files']
        total_training = cfg['loop_robot_data']['total_training']
        train_time = all_exp_files[:total_training]
        val_time = all_exp_files[total_training:]
        corresponding_class = corresponding()
        train_seq = corresponding_class.time_list_to_seq(train_time)
        print('Train seq:', train_seq)
        val_seq = corresponding_class.time_list_to_seq(val_time)
        print('Val seq:', val_seq)

        all_data_h5_files = sorted(glob.glob(join(data_dir, '*', '*.h5')))
        all_hallucination_files = sorted(glob.glob(join(hallucination_dir, '*', '*.h5')))

        training_files = []
        validation_files = []
        hallucination_train_files = []
        hallucination_val_files = []
        # select data h5 files based on train seq
        for train_seq_num in train_seq:
            key = 'seq_' + str(train_seq_num)
            for data_h5_file in all_data_h5_files:
                if key in data_h5_file:
                    training_files.append(data_h5_file)
                    break
            for hallucination_file in all_hallucination_files:
                if key in hallucination_file:
                    hallucination_train_files.append(hallucination_file)
                    break
        
        # select data h5 files based on val seq
        for val_seq_num in val_seq:
            key = 'seq_' + str(val_seq_num)
            for data_h5_file in all_data_h5_files:
                if key in data_h5_file:
                    validation_files.append(data_h5_file)
                    break
            for hallucination_file in all_hallucination_files:
                if key in hallucination_file:
                    hallucination_val_files.append(hallucination_file)
                    break
    else:
        raise ValueError('Invalid data type')
    
    
    # validation_files = sorted(glob.glob(join(data_dir, 'test', '*.h5')))
    # print(validation_files)

    # hallucination_val_files = sorted(glob.glob(join(hallucination_dir, 'val', '*.h5')))
    # print(join(hallucination_dir, 'val', '*.h5'))

    print('Training files:', training_files)
    print('Validation files:', validation_files)
    print('Hallucination training files:', hallucination_train_files)
    print('Hallucination validation files:', hallucination_val_files)

    x_thermal_val_1, x_thermal_val_2, x_imu_val_t, y_val_t, y_rgb_feat_val_t = odom_validation_stack_hallucination(validation_files,
                                                                                                                   hallucination_val_files,
                                                                                                                   sensor='thermal',
                                                                                                                   imu_length=IMU_LENGTH)
    len_val_i = y_val_t.shape[0]

    print('Final thermal validation shape:', np.shape(x_thermal_val_1), np.shape(y_val_t), np.shape(y_rgb_feat_val_t))

    # grap training files
    # training_files = sorted(glob.glob(join(data_dir, 'train', '*.h5')))
    n_training_files = len(training_files)
    # temp fix for training
    # start_idx = 12
    # training_file_idx = np.arange(start_idx, start_idx + n_training_files)
    seq_len = np.arange(n_training_files)

    for e in range(51):
        print("|-----> epoch %d" % e)
        # np.random.shuffle(training_files)
        random_idx = np.arange(n_training_files)
        np.random.shuffle(random_idx)
        training_files = [training_files[i] for i in random_idx]
        hallucination_train_files = [hallucination_train_files[i] for i in random_idx]
        for i in range(0, len(training_files)):

            # training_file = data_dir + '/train/' + data_type + '_seq_' + str(training_file_idx[seq_len[i]]) + '.h5'
            # hallucination_file = hallucination_dir + '/train/rgb_feat_seq_' + str(training_file_idx[seq_len[i]]) + '.h5'
            # print('---> Loading training file: turtle_seq_', str(training_file_idx[seq_len[i]]), '.h5',
                #   '---> Loading hallucinatio file: rgb_feat_seq_', str(training_file_idx[seq_len[i]]), '.h5')
            training_file = training_files[i]
            hallucination_file = hallucination_train_files[i]
            print('---> Loading training file:', training_file, '---> Loading hallucinatio file:', hallucination_file)

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
                        history = model.fit(x={'image_1': x_thermal_1, 'image_2': x_thermal_2, 'imu_data': x_imu},
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
                        history = model.fit({'image_1': x_thermal_1, 'image_2': x_thermal_2, 'imu_data': x_imu},
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
                    model.fit({'image_1': x_thermal_1, 'image_2': x_thermal_2, 'imu_data': x_imu},
                              {'time_distributed_1': y_label[:, :, 0:3], 'time_distributed_2': y_label[:, :, 3:6],
                               'flatten_rgb': y_rgb_feat},
                              batch_size=batch_size - 1, shuffle='batch', epochs=1, verbose=1)

        if ((e % 25) == 0):
            model.save(join(model_dir, str(e).format('h5')))

    print("Training for model has finished!")

    print('Saving training loss ....')
    train_loss = np.array(training_loss)
    loss_file_save = join(model_dir, 'training_loss.' + MODEL_NAME +'.h5')
    with h5py.File(loss_file_save, 'w') as hf:
        hf.create_dataset('train_loss', data=train_loss)

    print('Saving nn options ....')
    with open(join(model_dir, 'nn_opt.json'), 'w') as fp:
        json.dump(cfg['nn_opt']['tio_prob_params'], fp)

    print('Finished training ', str(training_files), ' trajectory!')

if __name__ == "__main__":
    os.system("hostname")
    set_seed(0)
    main()