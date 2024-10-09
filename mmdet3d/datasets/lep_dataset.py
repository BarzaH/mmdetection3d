import mmengine
import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.datasets.det3d_dataset import Det3DDataset


@DATASETS.register_module()
class MyDataset(Det3DDataset):

    METAINFO = {
        'classes': tuple(sorted(list(('LEP110_prom', 'vegetation', 'LEP110_anchor', 'power_lines', 'forest'))))
    }

    def parse_ann_info(self, info):
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        ann_info = anns_results = dict(
            gt_bboxes_3d=info['instances']['gt_bboxes_3d'],
            gt_labels_3d=np.array([self.METAINFO['classes'].index(class_name) for class_name in info['instances']['gt_names']]))
        if ann_info is None:
            ann_info = dict()
            # empty instance
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        # filter the gt classes not used in training
        ann_info = self._remove_dontcare(ann_info)
        gt_bboxes_3d = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'])
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info