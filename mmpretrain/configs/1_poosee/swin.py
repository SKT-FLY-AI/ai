_base_ = [
    '../_base_/models/swin_transformer_v2/base_256.py',
    '../_base_/datasets/imagenet_bs64_swin_256.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

data_root = '/root/ai/dataset/classification_aug_0816/'
model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-tiny-w8_3rdparty_in1k-256px_20220803-e318968f.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=7),
)

train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',       # We assume you are using the sub-folder format without ann_file
        data_prefix='train',    # The `data_root` is the data_prefix directly.
        classes=['1', '2', '3', '4', '5', '6', '7'],
        with_label=True,
    )
)

valid_dataloader = dict(
    batch_size=64,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',       # We assume you are using the sub-folder format without ann_file
        data_prefix='valid',    # The `data_root` is the data_prefix directly.
        classes=['1', '2', '3', '4', '5', '6', '7'],
        with_label=True,
    )
)

test_dataloader = dict(
    batch_size=64,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',       # We assume you are using the sub-folder format without ann_file
        data_prefix='test',    # The `data_root` is the data_prefix directly.
        classes=['1', '2', '3', '4', '5', '6', '7'],
        with_label=True,
    )
)

# runtime settings
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=5, save_best="auto"))
visualizer=dict(type='Visualizer', vis_backends=[dict(type='WandbVisBackend')])
randomness = dict(seed=0, diff_rank_seed=True)
find_unused_parameters=True
# auto resume
resume = True
load_from = '/root/ai/mmpretrain/work_dirs/swin_0818_aug/epoch_120.pth'
work_dir = "work_dirs/swin_0818_aug"
# logg  https://mmengine.readthedocs.io/en/latest/get_started/15_minutes.html