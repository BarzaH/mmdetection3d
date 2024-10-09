from typing import Callable, List, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes, get_box_type
from .det3d_dataset import Det3DDataset
import tempfile
import warnings
from os import path as osp

import torch
import mmcv
import numpy as np
from torch.utils.data import Dataset
from mmengine import print_log
from mmengine.fileio.file_client import FileClient
from terminaltables import AsciiTable
from mmengine.fileio import dump, list_from_file, load

from mmengine.dataset import Compose
from .utils import extract_result_dict, get_loading_pipeline


@DATASETS.register_module()
class Custom3DDataset(Dataset):
    """Customized 3D dataset.

    This is the base dataset of SUNRGB-D, ScanNet, nuScenes, and KITTI
    dataset.

    .. code-block:: none

    [
        {'sample_idx':
         'lidar_points': {'lidar_path': velodyne_path,
                           ....
                         },
         'annos': {'box_type_3d':  (str)  'LiDAR/Camera/Depth'
                   'gt_bboxes_3d':  <np.ndarray> (n, 7)
                   'gt_names':  [list]
                   ....
               }
         'calib': { .....}
         'images': { .....}
        }
    ]

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR'. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 file_client_args=dict(backend='disk')):
        super().__init__()
        self.data_root = data_root
        self.ann_file = osp.join(self.data_root, ann_file)
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

        self.CLASSES = self.get_classes(classes)
        self.file_client = FileClient(**file_client_args)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}

        # load annotations
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(self.ann_file) as local_path:
                self.data_infos = self.load_annotations(open(local_path, 'rb'))
        else:
            warnings.warn(
                'The used MMCV version does not have get_local_path. '
                f'We treat the {self.ann_file} as local paths and it '
                'might cause errors if the path is not a local path. '
                'Please use MMCV>= 1.3.16 if you meet errors.')
            self.data_infos = self.load_annotations(self.ann_file)

        # process pipeline
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # set group flag for the samplers
        if not self.test_mode:
            self._set_group_flag()

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        # loading data from a file-like object needs file format
        return load(ann_file, file_format='pkl')

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos["data_list"][index]
        sample_idx = info['sample_idx']
        pts_filename = osp.join(self.data_root, "points",
                                info['lidar_points']['lidar_path'])

        input_dict = dict(
            lidar_points=dict(lidar_path=pts_filename),
            sample_idx=sample_idx,
            file_name=pts_filename)

        annos = self.get_ann_info(index)
        if not self.test_mode:
            input_dict['ann_info'] = annos

            if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
                return None
        else:
            input_dict['eval_ann_info'] = annos
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos["data_list"][index]
        gt_bboxes_3d = info['instances']['gt_bboxes_3d']
        gt_names_3d = info['instances']['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # Obtain original box 3d type in info file
        ori_box_type_3d = info['instances']['box_type_3d']
        ori_box_type_3d, _ = get_box_type(ori_box_type_3d)

        # turn original box type to target box type
        gt_bboxes_3d = ori_box_type_3d(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['data_samples'].gt_instances_3d['labels_3d'] != -1).any()):
            return None
        return example

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: A list of class names.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results,
                tmp_dir is the temporal directory created for saving json
                files when ``jsonfile_prefix`` is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
            out = f'{pklfile_prefix}.pkl'
        dump(outputs, out)
        return outputs, tmp_dir

    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluate.

        Evaluation in indoor protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str], optional): Metrics to be evaluated.
                Defaults to None.
            iou_thr (list[float]): AP IoU thresholds. Defaults to (0.25, 0.5).
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        from mmdet3d.evaluation import indoor_eval
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'
        gt_annos = [info['instances'] for info in self.data_infos["data_list"]]
        label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}
        ret_dict = indoor_eval(
            gt_annos,
            results,
            iou_thr,
            label2cat,
            logger=logger,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d)
        if show:
            self.show(results, out_dir, pipeline=pipeline)

        return ret_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        raise NotImplementedError('_build_default_pipeline is not implemented '
                                  f'for dataset {self.__class__.__name__}')

    def _get_pipeline(self, pipeline):
        """Get data loading pipeline in self.show/evaluate function.

        Args:
            pipeline (list[dict]): Input pipeline. If None is given,
                get from self.pipeline.
        """
        if pipeline is None:
            if not hasattr(self, 'pipeline') or self.pipeline is None:
                warnings.warn(
                    'Use default pipeline for data loading, this may cause '
                    'errors when data is on ceph')
                return self._build_default_pipeline()
            loading_pipeline = get_loading_pipeline(self.pipeline.transforms)
            return Compose(loading_pipeline)
        return Compose(pipeline)

    def _extract_data(self, index, pipeline, key, load_annos=False):
        """Load data using input pipeline and extract data according to key.

        Args:
            index (int): Index for accessing the target data.
            pipeline (:obj:`Compose`): Composed data loading pipeline.
            key (str | list[str]): One single or a list of data key.
            load_annos (bool): Whether to load data annotations.
                If True, need to set self.test_mode as False before loading.

        Returns:
            np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:
                A single or a list of loaded data.
        """
        assert pipeline is not None, 'data loading pipeline is not provided'
        # when we want to load ground-truth via pipeline (e.g. bbox, seg mask)
        # we need to set self.test_mode as False so that we have 'annos'
        if load_annos:
            original_test_mode = self.test_mode
            self.test_mode = False
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = pipeline(input_dict)

        # extract data items according to keys
        if isinstance(key, str):
            data = extract_result_dict(example, key)
        else:
            data = [extract_result_dict(example, k) for k in key]
        if load_annos:
            self.test_mode = original_test_mode

        return data

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.data_infos["data_list"])

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

def eval_det_cls(pred, gt, iou_thr=None):
    # {img_id: {'bbox': box structure, 'det': matched list}}
    class_recs = {}
    npos = 0
    for img_id in gt.keys():
        cur_gt_num = len(gt[img_id])
        if cur_gt_num != 0:
            gt_cur = torch.zeros([cur_gt_num, 7], dtype=torch.float32)
            for i in range(cur_gt_num):
                gt_cur[i] = gt[img_id][i].tensor
            bbox = gt[img_id][0].new_box(gt_cur)
        else:
            bbox = gt[img_id]
        det = [[False] * len(bbox) for i in iou_thr]
        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det}

    # construct dets
    image_ids = []
    confidence = []
    ious = []
    for img_id in pred.keys():
        cur_num = len(pred[img_id])
        if cur_num == 0:
            continue
        pred_cur = torch.zeros((cur_num, 7), dtype=torch.float32)
        box_idx = 0
        for box, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            pred_cur[box_idx] = box.tensor
            box_idx += 1
        pred_cur = box.new_box(pred_cur)
        gt_cur = class_recs[img_id]['bbox']
        if len(gt_cur) > 0:
            # calculate iou in each image
            iou_cur = pred_cur.overlaps(pred_cur, gt_cur)
            for i in range(cur_num):
                ious.append(iou_cur[i])
        else:
            for i in range(cur_num):
                ious.append(np.zeros(1))

    confidence = np.array(confidence)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    image_ids = [image_ids[x] for x in sorted_ind]
    ious = [ious[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp_thr = [np.zeros(nd) for i in iou_thr]
    fp_thr = [np.zeros(nd) for i in iou_thr]
    for d in range(nd):
        R = class_recs[image_ids[d]]
        iou_max = -np.inf
        BBGT = R['bbox']
        cur_iou = ious[d]

        if len(BBGT) > 0:
            # compute overlaps
            for j in range(len(BBGT)):
                # iou = get_iou_main(get_iou_func, (bb, BBGT[j,...]))
                iou = cur_iou[j]
                if iou > iou_max:
                    iou_max = iou
                    jmax = j

        for iou_idx, thresh in enumerate(iou_thr):
            if iou_max > thresh:
                if not R['det'][iou_idx][jmax]:
                    tp_thr[iou_idx][d] = 1.
                    R['det'][iou_idx][jmax] = 1
                else:
                    fp_thr[iou_idx][d] = 1.
            else:
                fp_thr[iou_idx][d] = 1.

    ret = []
    for iou_idx, thresh in enumerate(iou_thr):
        # compute precision recall
        fp = np.cumsum(fp_thr[iou_idx])
        tp = np.cumsum(tp_thr[iou_idx])
        recall = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = average_precision(recall, precision)
        ret.append((recall, precision, ap))

    return ret


def eval_map_recall(pred, gt, ovthresh=None):
    ret_values = {}
    for classname in gt.keys():
        if classname in pred:
            ret_values[classname] = eval_det_cls(pred[classname],
                                                 gt[classname], ovthresh)
    recall = [{} for i in ovthresh]
    precision = [{} for i in ovthresh]
    ap = [{} for i in ovthresh]

    for label in gt.keys():
        for iou_idx, thresh in enumerate(ovthresh):
            if label in pred:
                recall[iou_idx][label], precision[iou_idx][label], ap[iou_idx][
                    label] = ret_values[label][iou_idx]
            else:
                recall[iou_idx][label] = np.zeros(1)
                precision[iou_idx][label] = np.zeros(1)
                ap[iou_idx][label] = np.zeros(1)

    return recall, precision, ap


def outdoor_eval(gt_annos,
                 dt_annos,
                 iou_thr,
                 label2cat,
                 logger=None,
                 box_type_3d=None,
                 box_mode_3d=None):
    assert len(dt_annos) == len(gt_annos)
    pred = {}  # map {class_id: pred}
    gt = {}  # map {class_id: gt}
    for img_id in range(len(dt_annos)):
        # parse detected annotations
        if 'pts_bbox' in dt_annos[img_id].keys():
            det_anno = dt_annos[img_id]['pts_bbox']
        else:
            det_anno = dt_annos[img_id]
        for i in range(len(det_anno['labels_3d'])):
            label = det_anno['labels_3d'].numpy()[i]
            bbox = det_anno['boxes_3d'].convert_to(box_mode_3d)[i]
            score = det_anno['scores_3d'].numpy()[i]
            if label not in pred:
                pred[int(label)] = {}
            if img_id not in pred[label]:
                pred[int(label)][img_id] = []
            if label not in gt:
                gt[int(label)] = {}
            if img_id not in gt[label]:
                gt[int(label)][img_id] = []
            pred[int(label)][img_id].append((bbox, score))

        # parse gt annotations
        gt_anno = gt_annos[img_id]
        if len(gt_anno['gt_bboxes_3d']) != 0:
            gt_boxes = box_type_3d(
                torch.from_numpy(gt_anno['gt_bboxes_3d']),
                box_dim=gt_anno['gt_bboxes_3d'].shape[-1],
                origin=(0.5, 0.5, 0.5)
            )
            labels_3d = torch.from_numpy(gt_anno['gt_labels_3d'])
        else:
            gt_boxes = box_type_3d(
                np.array([], dtype=np.float32),
                origin=(0.5, 0.5, 0.5))
            labels_3d = np.array([], dtype=np.int64)

        for i in range(len(labels_3d)):
            label = labels_3d[i].item()
            bbox = gt_boxes[i]
            if label not in gt:
                gt[label] = {}
            if img_id not in gt[label]:
                gt[label][img_id] = []
            gt[label][img_id].append(bbox)

    rec, prec, ap = eval_map_recall(pred, gt, iou_thr)
    ret_dict = dict()
    header = ['classes']
    table_columns = [[label2cat[label]
                      for label in ap[0].keys()] + ['Overall']]

    for i, iou_thresh in enumerate(iou_thr):
        header.append(f'AP_{iou_thresh:.2f}')
        header.append(f'AR_{iou_thresh:.2f}')
        rec_list = []
        for label in ap[i].keys():
            ret_dict[f'{label2cat[label]}_AP_{iou_thresh:.2f}'] = float(
                ap[i][label][0])
        ret_dict[f'mAP_{iou_thresh:.2f}'] = float(
            np.mean(list(ap[i].values())))

        table_columns.append(list(map(float, list(ap[i].values()))))
        table_columns[-1] += [ret_dict[f'mAP_{iou_thresh:.2f}']]
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

        for label in rec[i].keys():
            ret_dict[f'{label2cat[label]}_rec_{iou_thresh:.2f}'] = float(
                rec[i][label][-1])
            rec_list.append(rec[i][label][-1])
        ret_dict[f'mAR_{iou_thresh:.2f}'] = float(np.mean(rec_list))

        table_columns.append(list(map(float, rec_list)))
        table_columns[-1] += [ret_dict[f'mAR_{iou_thresh:.2f}']]
        table_columns[-1] = [f'{x:.4f}' for x in table_columns[-1]]

    table_data = [header]
    table_rows = list(zip(*table_columns))
    table_data += table_rows
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print_log('\n' + table.table, logger=logger)

    return ret_dict


@DATASETS.register_module()
class CustomDataset(Custom3DDataset):
    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos["data_list"][index]
        gt_bboxes_3d = info['instances']['gt_bboxes_3d']
        gt_names_3d = info['instances']['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d, dtype=np.int64)

        # Obtain original box 3d type in info file
        ori_box_type_3d = info['instances']['box_type_3d']
        ori_box_type_3d, _ = get_box_type(ori_box_type_3d)

        # turn original box type to target box type

        if len(gt_bboxes_3d) != 0:
            gt_bboxes_3d = ori_box_type_3d(
                gt_bboxes_3d,
                origin=(0.5, 0.5, 0.5),
                box_dim=gt_bboxes_3d.shape[-1]
            ).convert_to(self.box_mode_3d)
        else:
            # TODO: check that dim=9 is correct for all models
            gt_bboxes_3d = ori_box_type_3d(
                np.array([], dtype=np.float32),
                origin=(0.5, 0.5, 0.5),
                box_dim=7
            ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results

    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos["data_list"])
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'
        gt_annos = [info['instances'] for info in self.data_infos["data_list"]]
        label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}
        ret_dict = outdoor_eval(
            gt_annos,
            results,
            iou_thr,
            label2cat,
            logger=logger,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d)
        if show:
            self.show(results, out_dir, pipeline=pipeline)

        return ret_dict


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (np.ndarray): Recalls with shape of (num_scales, num_dets)
            or (num_dets, ).
        precisions (np.ndarray): Precisions with shape of
            (num_scales, num_dets) or (num_dets, ).
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or np.ndarray: Calculated average precision.
    """
    if recalls.ndim == 1:
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]

    assert recalls.shape == precisions.shape
    assert recalls.ndim == 2

    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')