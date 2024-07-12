custom_imports = dict(imports=[
    'mmdet3d.datasets.lep_dataset',
    'mmdet3d.datasets.transforms.formating',
    'mmdet3d.evaluation.metrics.outdoor_metric'
    ], allow_failed_imports=False)

dataset_type = 'MyDataset'
data_root = 'DATA_ROOT'
class_names = ['LEP110_anchor', 'LEP110_prom', 'power_lines', 'vegetation']
pcr = [
                -136.7853946685791, -135.2938232421875, -45.29965019226074,
                136.7853946685791, 135.2938232421875, 45.29965019226074
            ]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=pcr),
    dict(type='ObjectRangeFilter', point_cloud_range=pcr),
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
        use_dim=4),
    dict(type='Pack3DDetInputs', keys=['points'])
]

eval_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='Pack3DDetInputs', keys=['points']),
]
train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='custom_infos_train.pkl',
            data_prefix=dict(pts='points'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d='LiDAR')))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='points'),
        ann_file='custom_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))
val_evaluator = dict(
    type='OutdoorMetric',
    box_type_3d='LiDAR',
    classes=class_names,
    iou_thr=[0.25, 0.5])
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='points'),
        ann_file='custom_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR'))
test_evaluator = val_evaluator
