_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py',
    '../_base_/datasets/eddy_256x256.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
gpu_ids = range(1)
seed = 42
device = 'cuda'
work_dir = "/cta/users/emir/dev/ViT_Segmentation/conv_based_segmentors/output/"
model = dict(
    decode_head=dict(loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, avg_non_ignore=True, use_sigmoid=True),
        # dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
    ]))
