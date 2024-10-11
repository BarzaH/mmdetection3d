_base_ = [
    '../_base_/datasets/newcustom.py',
    '../_base_/models/centerpoint_sly.py',
    '../_base_/schedules/cosine.py', '../_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [
    -136.7853946685791, -135.2938232421875, -45.29965019226074,
    136.7853946685791, 135.2938232421875, 45.29965019226074
]
# Using calibration info convert the Lidar-coordinate point cloud range to the
# ego-coordinate point cloud range could bring a little promotion in nuScenes.
# point_cloud_range = [-51.2, -52, -5.0, 51.2, 50.4, 3.0]
# For nuScenes we usually do 10-class detection
# class_names = sorted(list(('LEP110_prom', 'vegetation', 'LEP110_anchor', 'power_lines', 'forest')))
data_prefix = dict(pts='', img='', sweeps='')

voxel_size = [0.18997971481747097, 0.18790808783637153, 2.264982509613037]
model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(point_cloud_range=point_cloud_range)),
    # pts_voxel_encoder=dict(point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])))

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=100)

# dataset_type = 'CustomDataset'
# data_root = 'DATA_ROOT'
# backend_args = None
#
# input_modality = dict(use_lidar=True, use_camera=False)
#
# train_pipeline = [
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=4,
#         use_dim=4,
#         backend_args=backend_args),
#     dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
#     dict(
#         type='GlobalRotScaleTrans',
#         rot_range=[-3.141592, 3.141592],
#         scale_ratio_range=[0.5, 1.5],
#         translation_std=[3.0, 3.0, 3.0]),
#     dict(
#         type='RandomFlip3D',
#         sync_2d=False,
#         flip_ratio_bev_horizontal=0.5,
#         flip_ratio_bev_vertical=0.5),
#     dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
#     dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
#     dict(type='PointShuffle'),
#     dict(
#         type='Pack3DDetInputs',
#         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
# ]
# test_pipeline = [
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=4,
#         use_dim=4,
#         backend_args=backend_args),
#     dict(
#         type='MultiScaleFlipAug3D',
#         img_scale=(1333, 800),
#         pts_scale_ratio=1,
#         flip=False,
#         transforms=[
#             dict(
#                 type='GlobalRotScaleTrans',
#                 rot_range=[0, 0],
#                 scale_ratio_range=[1., 1.],
#                 translation_std=[0, 0, 0]),
#             dict(type='RandomFlip3D'),
#             dict(
#                 type='PointsRangeFilter', point_cloud_range=point_cloud_range)
#         ]),
#     dict(type='Pack3DDetInputs', keys=['points'])
# ]
# eval_pipeline = [
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=4,
#         use_dim=4,
#         backend_args=backend_args),
#     dict(type='Pack3DDetInputs', keys=['points'])
# ]
#
# train_dataloader = dict(
#     batch_size=2,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='train.pkl',
#         pipeline=train_pipeline,
#         classes=class_names,
#         modality=input_modality,
#         test_mode=False,
#         box_type_3d='LiDAR',
#         file_client_args=dict(backend='disk')))
# test_dataloader = dict(
#     batch_size=2,
#     num_workers=1,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='val.pkl',
#         pipeline=test_pipeline,
#         classes=class_names,
#         modality=input_modality,
#         test_mode=True,
#         box_type_3d='LiDAR',
#         file_client_args=dict(backend='disk')))
# val_dataloader = dict(
#     batch_size=2,
#     num_workers=1,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='val.pkl',
#         pipeline=test_pipeline,
#         classes=class_names,
#         modality=input_modality,
#         test_mode=True,
#         box_type_3d='LiDAR',
#         file_client_args=dict(backend='disk')))
#
#
db_sampler = None