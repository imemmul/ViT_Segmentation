_base_ = [
    './_base_/models/seg_vit-b16.py',
    './_base_/datasets/eddy_256x256.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_20k.py'
]
in_channels = 768
img_size = 640
gpu_ids = range(1)
seed = 42
checkpoint = ""
device = 'cuda'
work_dir = "/cta/users/emir/dev/ViT_Segmentation/segViT/output/"
out_indices = [5, 7, 11]
model = dict(
    backbone=dict(
        img_size=(640, 640),
        embed_dims=768,
        num_layers=12,
        drop_path_rate=0.3,
        num_heads=12,
        out_indices=out_indices
        ),
    decode_head=dict(
        img_size=img_size,
        in_channels=in_channels,
        channels=in_channels,
        embed_dims=in_channels // 2,
        out_channels = 1,
        num_classes = 2,
        num_heads=12,
        use_stages=len(out_indices),
        loss_decode=dict(
            type='CrossEntropyLoss', avg_non_ignore=True, use_sigmoid=True), #use sigmoid
        ignore_index = 0
    ),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(608, 608)),
)

# jax use different img norm cfg
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
crop_size = (640, 640)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2048, 640)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 640),
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
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

optimizer = dict(type='AdamW', lr=0.003, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.),
                                                 'ln': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.),
                                                 }))
#
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=1e-5, by_epoch=False)


# model = dict(
#     backbone=dict(
#         img_size=(256, 256),
#         embed_dims=1024,
#         num_layers=24,
#         drop_path_rate=0.3,
#         num_heads=16,
#         # out_indices=out_indices
#         ),
#     decode_head=dict(
#         img_size=img_size,
#         in_channels=in_channels,
#         channels=in_channels,
#         embed_dims=in_channels // 2,
#         num_heads=16,
#         # use_stages=len(out_indices),
#         loss_decode=dict(
#             type='CrossEntropyLoss', use_sigmoid=True,), # to do use_mask=True  ====cross entropy loss instead atm loss===
#         threshold=0.6
#     ),
#     test_cfg=dict(mode='whole', crop_size=(256, 256), stride=(208, 208)), # dont forget to update stride in seg_vit_b16.py
# )