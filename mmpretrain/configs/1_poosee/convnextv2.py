_base_ = [
    '../_base_/models/convnext_v2/nano.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

data_root = '/root/ai/dataset/puppy_poo/dataset_cls_train_aug_apply'


train_dataloader = dict(
    batch_size=16,
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
    batch_size=16,
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
    batch_size=16,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',       # We assume you are using the sub-folder format without ann_file
        data_prefix='test',    # The `data_root` is the data_prefix directly.
        classes=['1', '2', '3', '4', '5', '6', '7'],
        with_label=True,
    )
)
# schedule setting
optim_wrapper = dict(
    optimizer=dict(lr=8e-4, weight_decay=0.3),
    clip_grad=None,
)

# learning policy
param_scheduler = [dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True)]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]

default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=5, save_best="auto"))
visualizer=dict(type='Visualizer', vis_backends=[dict(type='WandbVisBackend')])

resume = False

work_dir = "work_dirs/cnextv2_seg_aug"

