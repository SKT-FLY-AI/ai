_base_ = [
    '../_base_/models/resnet101.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]


data_root = "/root/ai/dataset/puppy_poo/dataset_cls_train_aug_apply_retype"

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
work_dir = "work_dirs/resnet101_8xb32"
# logg  https://mmengine.readthedocs.io/en/latest/get_started/15_minutes.html