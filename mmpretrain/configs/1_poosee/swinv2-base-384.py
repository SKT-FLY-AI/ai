_base_ = [
    '../_base_/models/swin_transformer_v2/base_384.py',
    '../_base_/datasets/imagenet_bs64_swin_384.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

data_root = '/root/ai/dataset/classification_aug_apply/'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        img_size=384,
        window_size=[24, 24, 24, 12],
        drop_path_rate=0.2,
        pretrained_window_sizes=[12, 12, 12, 6]))

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

# runtime settings
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=5, save_best="auto"))
visualizer=dict(type='Visualizer', vis_backends=[dict(type='WandbVisBackend')])
# auto resume
resume = False
work_dir = "work_dirs/swin384"