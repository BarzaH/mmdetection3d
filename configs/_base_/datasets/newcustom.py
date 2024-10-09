custom_imports = dict(imports=[
    'mmdet3d.datasets.custom_dataset',
    'mmdet3d.datasets.transforms.formating',
    'mmdet3d.models.detectors',
    'mmdet3d.models.dense_heads',
    'mmdet3d.evaluation.metrics.outdoor_metric'], allow_failed_imports=False)

point_cloud_range = [
    -136.7853946685791, -135.2938232421875, -45.29965019226074,
    136.7853946685791, 135.2938232421875, 45.29965019226074
]
class_names = sorted(list(('LEP110_prom', 'vegetation', 'LEP110_anchor', 'power_lines', 'forest')))
dataset_type='CustomDataset'
data_root='DATA_ROOT'

input_modality = dict(use_lidar=True, use_camera=False)
data_prefix = dict(pts='bin', img='', sweeps='')
backend_args = None

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.141592, 3.141592],
        scale_ratio_range=[0.5, 1.5],
        translation_std=[3.0, 3.0, 3.0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='custom_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR',
        file_client_args=dict(backend='disk')))
test_dataloader = dict(
    batch_size=16,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='custom_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        file_client_args=dict(backend='disk')))
val_dataloader = dict(
    batch_size=16,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='custom_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        file_client_args=dict(backend='disk')))

val_evaluator = dict(
    type='OutdoorMetric',
    box_type_3d='LiDAR',
    classes=class_names,
    iou_thr=[0.25, 0.5])
test_evaluator = val_evaluator