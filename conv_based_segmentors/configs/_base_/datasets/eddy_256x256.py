# dataset settings
dataset_type = 'EddyDatasetREGISTER'
data_root = "/cta/users/emir/dev/datasets/eddy_dataset/"
train_dir = "train_data_aug/"
train_annot= "train_label_aug/"
valid_dir = "valid_data_aug/"
valid_annot = "valid_label_aug/"
# data_root = '/home/emir/Desktop/dev/myResearch/dataset/dataset_eddy/data4test/'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)

crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 256),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=train_dir,
        ann_dir=train_annot,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=valid_dir,
        ann_dir=valid_annot,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=valid_dir,
        ann_dir=valid_annot,
        pipeline=test_pipeline))
