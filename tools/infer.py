import argparse
import logging
import os
import os.path as osp
import json

import torch
import numpy as np

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.apis import inference_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Inference a 3D detector')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='Model weights path')
    parser.add_argument('input_path', help='Path to directory or .pcd.bin file')
    parser.add_argument('save_path', help='Where to store detection results')
    return parser.parse_args()

def tensor_to_list(tensor):
    return tensor.cpu().numpy().tolist()

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
    cfg.load_from = args.checkpoint
    runner = Runner.from_cfg(cfg)
    runner.model.cfg = runner.cfg

    os.makedirs(args.save_path, exist_ok=True)
    if osp.isfile(args.input_path):
        input_files = [args.input_path]
    else:
        input_files = [osp.join(args.input_path, f)
                       for f in os.listdir(args.input_path) if f.endswith('bin')]
    logging.info(f'Found files: {input_files}')
    for input_file in input_files:
        try:
            result = inference_detector(runner.model, input_file)[0]
        except:
            logging.info(f'Could not infer {input_file}')
        input_filename = osp.basename(result.lidar_path)
        bboxes_3d = result.pred_instances_3d.bboxes_3d  # torch.tensor of shape (N, 7)
        scores_3d = result.pred_instances_3d.scores_3d  # torch.tensor of shape (N,)
        labels_3d = result.pred_instances_3d.labels_3d  # torch.tensor of shape (N,)

        result_dict = {
            'input_filename': input_filename,
            'bboxes_3d': tensor_to_list(bboxes_3d),
            'scores_3d': tensor_to_list(scores_3d),
            'labels_3d': tensor_to_list(labels_3d)
        }

        output_filename = osp.splitext(input_filename)[0] + '_results.json'
        output_path = osp.join(args.save_path, output_filename)
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)

    print(f"Results saved to {args.save_path}")

if __name__ == '__main__':
    main()

