import yaml
import shutil
import subprocess
import os
import shutil
import subprocess
import zipfile

# model_names = ['neural_odometry_v2_new_data',
#                'neural_odometry_v2_new_data_3c_5e_10r',
#                'neural_odometry_v2_new_data_3c_5e_10r_fedtrim',
#                'neural_odometry_v2_new_data_6c_5e_10r',
#                'neural_odometry_v2_new_data_6c_5e_10r_fedtrim']

model_names = ['neural_odometry_v2_single_fp16',]

# Step 1: Read the config.yaml file
with open('config_auto.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Step 3: Update model_name and run embeeding_test.sh for each model name
for model_name in model_names:
    result_folder = './results'
    if os.path.exists(result_folder):
        shutil.rmtree(result_folder)

    # Update the model_name in config_auto.yaml
    config['os_odom_test']['model'] = model_name
    with open('config_auto.yaml', 'w') as file:
        yaml.safe_dump(config, file)
    
    subprocess.run([
        'bash', '-c', 
        'export CUDA_VISIBLE_DEVICES=0 && /mnt/data/hantaozhong/conda_env/ti-slam/bin/python os_test_odom_auto.py'
    ])

   
    zip_filename = f"{model_name}_odom.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, dirs, files in os.walk(result_folder):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, result_folder))