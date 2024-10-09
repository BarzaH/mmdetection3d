import argparse
import logging
import os
import os.path as osp
import json
from pathlib import Path

from mmengine.config import Config
from mmdet3d.apis import init_model
import torch
import open3d as o3d
import numpy as np
import plotly.graph_objects as go
from mmdet3d.apis import inference_detector
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion

class BoundingBox:
    """ Simple data class representing a 3d box including, label, score and velocity. """

    def __init__(self, center, size, orientation, velocity=(np.nan, np.nan, np.nan)):
        """
        :param center: [<float>: 3]. Center of box given as x, y, z.
        :param size: [<float>: 3]. Size of box in width, length, height.
        :param orientation: <Quaternion>. Box orientation.
        :param velocity: [<float>: 3]. Box velocity in x, y, z direction.
        """
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.velocity = np.array(velocity)

    def corners(self, wlh_factor=1.0):
        """
        Returns the bounding box corners.
        :param wlh_factor: <float>. Multiply w, l, h by a factor to inflate or deflate the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, l, h = self.wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

def get_centrize_transform_vals(x,y,z):
    x_transformation = - (x.min() + (x.max() - x.min()) * 0.5)
    y_transformation = - (y.min() + (y.max() - y.min()) * 0.5)
    z_transformation = - (z.min() + (z.max() - z.min()) * 0.5)
    return x_transformation, y_transformation, z_transformation

def extract_xyz(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return x, y, z


def rotate(source_angle, delta):
    result = source_angle + delta
    if result > np.pi:
        result = -np.pi + (result - np.pi)
    elif result < -np.pi:
        result = np.pi + (result + np.pi)
    return result


def get_slide_boxes(pointcloud_range, model_pcr_dim, apply_sw):
    pcd = pointcloud_range.copy()
    ws = model_pcr_dim.copy()
    if isinstance(apply_sw, dict):
        apply_sw = list(apply_sw.values())
    slides = [0, 0, 0]
    for i in range(3):
        if apply_sw is not None and not apply_sw[i]:
            slides[i] = 1
            continue
        slides[i], overlap = divmod(pcd[i], ws[i])
        if overlap > 0:
            slides[i] += 1

    sboxes = []
    for z in range(int(slides[2])):
        for y in range(int(slides[1])):
            for x in range(int(slides[0])):
                sboxes.append([
                    ws[0] * x,
                    ws[0] * (x + 1),
                    ws[1] * y,
                    ws[1] * (y + 1),
                    ws[2] * z,
                    ws[2] * (z + 1),
                ])
    return sboxes


def get_per_box_predictions(result, score_thr, selected_classes, cfg, center_vec, input_slide_range):
    preds = result.pred_instances_3d

    labels = cfg["class_names"]
    gt_index_to_labels = dict(enumerate(labels))
    model_name = "CenterPoint"

    pred_scores = preds['scores_3d'].cpu().numpy()
    pred_bboxes = preds['bboxes_3d'].tensor.cpu().numpy()  # x, y, z, dx, dy, dz, rot, vel_x, vel_y
    pred_labels = preds['labels_3d'].cpu().numpy()

    inds = pred_scores > score_thr
    pred_bboxes = pred_bboxes[inds]
    pred_labels = pred_labels[inds]
    pred_scores = pred_scores[inds]

    assert len(pred_bboxes) == len(pred_scores) == len(pred_labels)
    results = []
    for i in range(len(pred_bboxes)):
        det = {}
        det["detection_name"] = gt_index_to_labels[pred_labels[i]]
        if selected_classes is not None and det["detection_name"] not in selected_classes:
            continue
        det["size"] = pred_bboxes[i, 3:6].tolist()
        if cfg.dataset_type != "SuperviselyDataset":
            det["size"] = [det["size"][1], det["size"][0], det["size"][2]]
        det["translation"] = pred_bboxes[i, :3].tolist()
        # TODO: check these conditions in the future
        if cfg.dataset_type != "SuperviselyDataset":
            det["translation"][2] += det["size"][2] * 0.5
        if model_name == "CenterPoint":
            det["translation"][2] += det["size"][2] * 0.5

        for k in range(3):
            det["translation"][k] += center_vec[k]
        # skip boxes out of pointcloud range
        if det["translation"][0] < input_slide_range[0] or \
                det["translation"][0] > input_slide_range[3] or \
                det["translation"][1] < input_slide_range[1] or \
                det["translation"][1] > input_slide_range[4] or \
                det["translation"][2] < input_slide_range[2] or \
                det["translation"][2] > input_slide_range[5]:
            continue
        det["rotation"] = pred_bboxes[i, 6].item()
        if cfg.dataset_type != "SuperviselyDataset":
            det["rotation"] = rotate(det["rotation"], -np.pi * 0.5)
        det["velocity"] = pred_bboxes[i, 7:].tolist()
        det["detection_score"] = pred_scores[i].item()
        results.append(det)
    return results


def inference_model(model, local_pointcloud_path, thresh=0.233, selected_classes=None,
                    apply_sw=None, center_ptc=None):
    if isinstance(center_ptc, dict):
        center_ptc = list(center_ptc.values())
    pcd = o3d.io.read_point_cloud(local_pointcloud_path)
    pcd_np = np.asarray(pcd.points)
    # check ptc ranges
    pcr = model.cfg.point_cloud_range
    pcr_dim = [pcr[3] - pcr[0], pcr[4] - pcr[1], pcr[5] - pcr[2]]
    input_ptc_dim = [
        pcd_np[:, 0].max() - pcd_np[:, 0].min(),
        pcd_np[:, 1].max() - pcd_np[:, 1].min(),
        pcd_np[:, 2].max() - pcd_np[:, 2].min()
    ]

    sboxes = get_slide_boxes(input_ptc_dim, pcr_dim, apply_sw)

    pcd_sboxes = []
    for sbox in sboxes:
        pcd_sbox = []
        for i in range(3):
            if center_ptc is None or center_ptc[i]:
                pcd_sbox.extend([
                    pcd_np[:, i].min() + sbox[i * 2],
                    pcd_np[:, i].min() + sbox[i * 2 + 1]
                ])
            else:
                pcd_sbox.extend([
                    pcr[i],
                    pcr[i + 3]
                ])
        pcd_sboxes.append(pcd_sbox)

    results = []
    # TODO: is it possible to use batch inference here?
    for sbox in pcd_sboxes:
        pcd_eps = 1e-3
        pcd_slide = pcd_np[
            (pcd_np[:, 0] > sbox[0] - pcd_eps) &
            (pcd_np[:, 0] < sbox[1] + pcd_eps) &
            (pcd_np[:, 1] > sbox[2] - pcd_eps) &
            (pcd_np[:, 1] < sbox[3] + pcd_eps) &
            (pcd_np[:, 2] > sbox[4] - pcd_eps) &
            (pcd_np[:, 2] < sbox[5] + pcd_eps)
            ]
        if len(pcd_slide) == 0:
            continue
        center_vec = [0, 0, 0]
        input_slide_range = [
            pcd_slide[:, 0].min(),
            pcd_slide[:, 1].min(),
            pcd_slide[:, 2].min(),
            pcd_slide[:, 0].max(),
            pcd_slide[:, 1].max(),
            pcd_slide[:, 2].max()
        ]
        for i in range(3):
            if center_ptc is None or center_ptc[i]:
                dim_trans = input_slide_range[i] + (input_slide_range[i + 3] - input_slide_range[i]) * 0.5
                pcd_slide[:, i] -= dim_trans
                center_vec[i] = dim_trans

        intensity = np.zeros((pcd_slide.shape[0], 1), dtype=np.float32)
        pcd_slide = np.hstack((pcd_slide, intensity))

        result, _ = inference_detector(model, pcd_slide.astype(np.float32))
        result = get_per_box_predictions(result, thresh, selected_classes, model.cfg, center_vec, input_slide_range)
        results.extend(result)
    return results


def get_box_vertexes(center, size, rotation):
    rot = Rotation.from_rotvec(rotation)
    rot_mat = rot.as_matrix()
    orientation = Quaternion(matrix=rot_mat)
    bbox = BoundingBox(center, size, orientation)
    return bbox.corners()


def visualize_pcd_with_bboxes(results, local_pointcloud_path, output_path):
    pcd_x, pcd_y, pcd_z = extract_xyz(local_pointcloud_path)
    x_transformation, y_transformation, z_transformation = get_centrize_transform_vals(pcd_x, pcd_y, pcd_z)
    boxes_data = []
    for result in results:
        box_x, box_y, box_z = get_box_vertexes(
            center=result["translation"] + np.array([x_transformation, y_transformation, z_transformation]),
            size=result["size"],
            rotation=[0, 0, result["rotation"] + (np.pi / 2)],
        )
        box_viz = go.Mesh3d(
            x=box_x,
            y=box_y,
            z=box_z,
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            opacity=0.3,
            color='#DC143C',
            flatshading=True
        )
        boxes_data.append(box_viz)

    boxes_data.insert(0, go.Scatter3d(
        x=pcd_x + x_transformation,
        y=pcd_y + y_transformation,
        z=pcd_z + z_transformation,
        mode='markers',
        marker=dict(
            size=2,
            opacity=0.8
        )))

    # visualize inference results
    fig = go.Figure(data=boxes_data)
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
    fig.write_html(output_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Inference a 3D detector')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='Model weights path')
    parser.add_argument('input_path', help='Path to directory or .pcd.bin file')
    parser.add_argument('save_path', help='Where to store detection results')
    parser.add_argument(
        '--data_root',
        default=None,
        help='Input data dir')
    parser.add_argument(
        '--class_names',
        default=None,
        help='List of class names')
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.data_root:
        cfg.data_root = args.data_root
        cfg.train_dataloader.dataset.data_root = args.data_root
        cfg.val_dataloader.dataset.data_root = args.data_root
        cfg.test_dataloader.dataset.data_root = args.data_root
    if args.class_names:
        cfg.class_names = eval(args.class_names)
        cfg.train_dataloader.dataset.classes = eval(args.class_names)
        cfg.val_dataloader.dataset.classes = eval(args.class_names)
        cfg.test_dataloader.dataset.classes = eval(args.class_names)
        cfg.val_evaluator.classes = eval(args.class_names)
        cfg.test_evaluator.classes = eval(args.class_names)

        cfg.model.pts_bbox_head.tasks = [dict(num_class=1, class_names=[class_name]) for class_name in eval(args.class_names)]


    cfg.work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
    cfg.load_from = args.checkpoint

    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    model = init_model(cfg,
                       cfg.load_from,
                       device)


    os.makedirs(args.save_path, exist_ok=True)


    if osp.isfile(args.input_path):
        input_files = [args.input_path]
    else:
        input_path = args.input_path
        if not str(input_path).endswith("pointcloud"):
            input_path = osp.join(input_path, "pointcloud")
        input_files = [osp.join(input_path, f)
                       for f in os.listdir(input_path) if f.endswith('.pcd')]


    logging.info(f'Found files: {input_files}')
    for input_file in input_files:
        try:
            input_filename = osp.basename(input_file)
            output_filename = osp.splitext(input_filename)[0] + '_results.json'
            output_path = osp.join(args.save_path, output_filename)
            results = inference_model(model, input_file)
            visualize_pcd_with_bboxes(results, input_file, output_path.replace("json", "html"))
            result_dict = {
                'input_filename': input_filename,
                'results': results
            }

            with open(output_path, 'w') as f:
                json.dump(result_dict, f, indent=2)
        except:
            logging.info(f'Could not infer {input_file}')
            continue


    print(f"Results saved to {args.save_path}")

if __name__ == '__main__':
    main()

