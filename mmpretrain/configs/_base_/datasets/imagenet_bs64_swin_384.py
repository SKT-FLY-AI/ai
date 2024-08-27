# dataset settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=7,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=384,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=384, backend='pillow', interpolation='bicubic'),
    dict(type='PackInputs'),
]

data_root = '/root/ai/dataset/classification_aug_apply/'
train_dataloader = dict(
    batch_size=16,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',       # We assume you are using the sub-folder format without ann_file
        data_prefix='train',    # The `data_root` is the data_prefix directly.
        classes=['1', '2', '3', '4', '5', '6', '7'],
        with_label=True,
        pipeline=train_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',       # We assume you are using the sub-folder format without ann_file
        data_prefix='train',    # The `data_root` is the data_prefix directly.
        classes=['1', '2', '3', '4', '5', '6', '7'],
        with_label=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
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
        pipeline=test_pipeline
    )
)

val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
# test_dataloader = val_dataloader
test_evaluator = val_evaluator
