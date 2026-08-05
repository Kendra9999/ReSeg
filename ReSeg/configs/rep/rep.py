_base_ = '../_base_/default_runtime.py'

model = dict(
    type='Rep',
    backbone=dict(
        type='ResNet3d',
        pretrained2d=True,
        pretrained='torchvision://resnet18',
        depth=18,
        in_channels=1,
        spatial_strides=(2, 2, 2, 2),
        temporal_strides=(2, 2, 2, 2), #(1, 1, 1, 2),
        conv1_kernel=(7, 7, 7), #(3, 7, 7),
        conv1_stride_t=1,
        conv1_stride_s=1,
        pool1_stride_t=2, #1,
        pool1_stride_s=2,
        with_pool1=False,
        with_pool2=False, #True,
        conv_cfg=dict(type='Conv3d'),
        inflate=((1, 1), (1, 1), (1, 1), (1, 1)), #((0, 0), (0, 0), (1, 1), (1, 1)),
        # norm_cfg = dict(type='GN',num_groups=32, requires_grad=True),
        norm_eval=False,
        zero_init_residual=False),
    neck=dict(
        type='FPN3d',
        start_level=0,
        end_level=4,
        in_channels=[64, 128, 256, 512],
        out_channels=128,
        num_outs=4,
        conv_cfg=dict(type='Conv3d')),
    patchloss=dict(type='SupPatchNCELoss',
                   temperature=0.33,
                   ),
    superloss=dict(type='SupConMeanLoss',
                   temperature=0.1,
                   base_temperature=1.0,
                   ),
    # model training and testing settings
    train_cfg=dict(
        intra_cfg=dict(
            select_anchor_number=256,
            select_other_number=512),
    ),
    test_cfg=dict(
        save_path='/data1/ydr/Project_representation/SSL_contrastive/work_dirs/results/',
        output_embedding=True
    ))


synthetic_data_root = "/mnt/sdb/drong/Project_representation/Data_gen/Totalsegmentator_gen_v1/"

overlap_patch_set = dict(
    type = 'Dataset3dSynthetic',
    data_dir = synthetic_data_root,
    crop_transform = True,
    crop_transform_kwargs = dict(
        crop_size = (96, 96, 96),
    )
)

whole_image_set = dict(
    type = 'Dataset3dSynthetic',
    data_dir = synthetic_data_root,
)

data = dict(
    samples_per_gpu=3, #5,
    workers_per_gpu=12,
    train=dict(
        type='ConcatDataset',
        datasets=[overlap_patch_set, whole_image_set]
    ),
    val=dict(),
    test=dict(), 
)


find_unused_parameters = True

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=1000,
    gamma=0.95)
runner = dict(type="IterBasedRunner", max_iters=45000)
fp16 = dict(loss_scale="dynamic")


checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=20)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project = "SSL_contrastive",
                entity = "1820037839-shanghai-jiao-tong-university",
                name = "rep",
            ),
        )
    ])