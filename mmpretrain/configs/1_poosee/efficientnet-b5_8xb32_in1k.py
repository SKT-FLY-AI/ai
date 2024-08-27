_base_ = [
    '../_base_/models/efficientnet_b5.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]
# pretrained = "https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b5_3rdparty-ra-noisystudent_in1k_20221103-111a185f.pth"
# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetRandomCrop', scale=456),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetCenterCrop', crop_size=456),
    dict(type='PackInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# runtime settings
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=5, save_best="auto"))
visualizer=dict(type='Visualizer', vis_backends=[dict(type='WandbVisBackend')])
randomness = dict(seed=0, diff_rank_seed=True)
find_unused_parameters=True
# auto resume
resume = False
# load_from = '/root/ai/mmpretrain/work_dirs/swin_0818_aug/epoch_120.pth'
work_dir = "work_dirs/effib5_type4"