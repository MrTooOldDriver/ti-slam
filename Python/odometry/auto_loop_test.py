import yaml
import shutil
import subprocess
import os
import shutil
import subprocess
import zipfile


# embedding, model_name = 'neural_embedding_v2_new_data_single', 'neural_loop_closure_v2_new_data'
embedding, model_name = 'neural_embedding_v2_single_fp16', 'neural_loop_closure_v2_single_fp16'
# embedding, model_name = 'neural_embedding_v2_single_fp32', 'neural_loop_closure_v2_single_fp32'
# embedding, model_name = 'neural_embedding_v2_new_data_3c5e10r_fedtrim', 'neural_loop_closure_v2_new_data_3c5e10r_fedtrim'
# embedding, model_name = 'neural_embedding_v2_new_data_6c5e10r', 'neural_loop_closure_v2_new_data_6c5e10r'
# embedding, model_name = 'neural_embedding_v2_new_data_6c5e10r_fedtrim', 'neural_loop_closure_v2_new_data_6c5e10r_fedtrim'
result_folder = './results'

# Step 1: Read the config.yaml file
with open('config_auto.yaml', 'r') as file:
    config = yaml.safe_load(file)


config['loop_evaluation_opt']['loop_detection_network'] = embedding
config['loop_evaluation_opt']['model_name'] = model_name
with open('config_auto.yaml', 'w') as file:
    yaml.safe_dump(config, file)

subprocess.run([
    'bash', '-c', 
    'export CUDA_VISIBLE_DEVICES=0 && /mnt/data/hantaozhong/conda_env/ti-slam/bin/python os_test_loop_auto.py'
])

zip_filename = f"{model_name}_loop.zip"
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for root, dirs, files in os.walk(result_folder):
        for file in files:
            file_path = os.path.join(root, file)
            zipf.write(file_path, os.path.relpath(file_path, result_folder))

