import argparse
import logging
import os
import os.path as osp
import json

import torch
import numpy as np


import open3d as o3d
import plotly.graph_objects as go

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.fileio import get

from mmdet3d.apis import inference_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Inference a 3D detector')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='Model weights path')
    parser.add_argument('input_path', help='Path to directory or .pcd.bin file')
    parser.add_argument('save_path', help='Where to store detection results')
    return parser.parse_args()

def visualize_pcd(pcd_x, pcd_y, pcd_z, bboxes, plot_name):
    fig = go.Figure(data=[
        go.Scatter3d(
            x=pcd_x,
            y=pcd_y,
            z=pcd_z,
            mode='markers',
            marker=dict(
                size=2,
                opacity=0.8
            ))])

    for bbox in bboxes:
        x, y, z, x_size, y_size, z_size, yaw = bbox
        corners = np.array([
            [x - x_size / 2, y - y_size / 2, z - z_size / 2],
            [x + x_size / 2, y - y_size / 2, z - z_size / 2],
            [x + x_size / 2, y + y_size / 2, z - z_size / 2],
            [x - x_size / 2, y + y_size / 2, z - z_size / 2],
            [x - x_size / 2, y - y_size / 2, z + z_size / 2],
            [x + x_size / 2, y - y_size / 2, z + z_size / 2],
            [x + x_size / 2, y + y_size / 2, z + z_size / 2],
            [x - x_size / 2, y + y_size / 2, z + z_size / 2]
        ])

        # Rotate the corners based on the yaw angle
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        corners = np.dot(corners, rotation_matrix)

        edges = np.array([
            [corners[0], corners[1]],
            [corners[1], corners[2]],
            [corners[2], corners[3]],
            [corners[3], corners[0]],
            [corners[4], corners[5]],
            [corners[5], corners[6]],
            [corners[6], corners[7]],
            [corners[7], corners[4]],
            [corners[0], corners[4]],
            [corners[1], corners[5]],
            [corners[2], corners[6]],
            [corners[3], corners[7]]
        ])
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=edge[:, 0],
                y=edge[:, 1],
                z=edge[:, 2],
                mode='lines',
                line=dict(color='red', width=2)
            ))

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(range=[-100, 100]),
            yaxis=dict(range=[-80, 80]),
            zaxis=dict(range=[-30, 30])
        ),
    )
    fig.write_html(f"{plot_name}.html")

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
            continue
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
        pts_bytes = get(input_file)
        points = np.frombuffer(pts_bytes, dtype=np.float32)
        points = points.reshape(-1, 4)
        pcd_x, pcd_y, pcd_z = points[:10000, 0], points[:10000, 1], points[:10000, 2]
        visualize_pcd(pcd_x, pcd_y, pcd_z, bboxes_3d[:10].cpu().numpy(), output_path.replace('.json', ''))

    print(f"Results saved to {args.save_path}")

if __name__ == '__main__':
    main()

