import glob
from os.path import join
from time_sqe_corresponding import corresponding

def process_turtle_data(data_type, cfg, data_dir, hallucination_dir):
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
    return training_files, validation_files, hallucination_train_files, hallucination_val_files
