import os
import yaml
from os.path import join
import multiprocessing
import inspect


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

def run_command(eval_exp, model, epoch, dataroot, loop_pairs_path, net_name, thres, img_h, img_w, img_c, gpu_id):
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python -W ignore utility/test_loop_pose_prob.py --eval_exp {eval_exp} --model {model} --epoch {epoch} --dataroot {dataroot} --loop_path {loop_pairs_path} --net_name {net_name} --thres {thres} --img_h {img_h} --img_w {img_w} --img_c {img_c}'
    os.system(cmd)

def main():
    # === Load configuration and list of training data ===
    with open(join(currentdir, 'config_auto.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)

    # Evaluation Parameters
    model = cfg['loop_evaluation_opt']['model_name']
    evaluation_experiments = cfg['loop_evaluation_opt']['exp_files']

    dataroot = cfg['loop_evaluation_opt']['dataroot']
    loop_pairs_path = cfg['loop_evaluation_opt']['loop_pairs_path']
    net_name = cfg['loop_evaluation_opt']['loop_detection_network']
    thres = cfg['loop_evaluation_opt']['loop_threshold']
    img_h = cfg['nn_opt']['loop_params']['img_h']
    img_w = cfg['nn_opt']['loop_params']['img_w']
    img_c = cfg['nn_opt']['loop_params']['img_c']

    # Evaluate the model we defined
    print('Test Model {}'.format(model))
    epochs = ['best']
    print(epochs)

    # Generate the output
    commands = []
    for i, eval_exp in enumerate(evaluation_experiments):
        for epoch in epochs:
            gpu_id = i % 9 + 4 # Cycle through GPU IDs 0 to 8
            commands.append((eval_exp, model, epoch, dataroot, loop_pairs_path, net_name, thres, img_h, img_w, img_c, gpu_id))

    # Run commands in parallel
    with multiprocessing.Pool(processes=9) as pool:
        pool.starmap(run_command, commands)

if __name__ == "__main__":
    os.system("hostname")
    main()