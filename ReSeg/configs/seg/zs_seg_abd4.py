_base_ = '../_base_/default_runtime.py'

num_classes = 4
patch_size = (128, 128, 128)

model = dict(
    type='ZS_Seg',
    pretrained_model = dict(
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
        )),
    pretrained_checkpoint = 'work_dirs/works/rep/20260420_160959/iter_45000.pth',
    seg_decoder=dict(
        type='SegDecoder',
        in_channels=[64, 64, 128, 256, 512],
        aux_channels=256,
        num_classes=num_classes,
        deep_supervision=True,
    ),
    patch_size = patch_size,
    segloss=dict(
        type='DeepSupervisionWrapper', 
        loss=dict(
            type='DC_and_CE_loss',
            soft_dice_kwargs = dict(batch_dice=True, smooth=1e-5, do_bg=False),
            ce_kwargs = dict(),
            weight_ce=1, weight_dice=1,
        ),
        weight_factors=[0.53333333, 0.26666667, 0.13333333, 0.06666667],
    )
)

synthetic_seg_data_root = "/mnt/sdb/drong/Project_representation/Data_gen/Totalsegmentator_gen_seg/"
seg_label_dir = "labels_abd4"
seg_label_dict_json = "seg_labels_abd4.json"

transform_kwargs = dict(
    patch_size = patch_size,
    spacing = (3., 3., 3.),
    sw_batch_size = 1,
)

BCV_image_dir = "/mnt/sdb/drong/Project_nnUNet/DATASET/nnUNet_raw/Dataset008_BCV_abd4/imagesTr/"
BCV_label_dir = "/mnt/sdb/drong/Project_nnUNet/DATASET/nnUNet_raw/Dataset008_BCV_abd4/labelsTr/"

data = dict(
    samples_per_gpu=2, 
    workers_per_gpu=12,
    train=dict(
        type='Dataset3dSynSeg',
        data_dir = synthetic_seg_data_root,
        transform_kwargs = transform_kwargs,
        seg_label_dir = seg_label_dir,
        seg_label_dict_json = seg_label_dict_json,
    ),
    val=dict(
        type='ValidationDataset',
        image_dir = BCV_image_dir,
        label_dir = BCV_label_dir,
        num_classes = num_classes,
        patch_size = transform_kwargs["patch_size"],
        spacing = transform_kwargs["spacing"],
    ),
    test=dict(), 
)

evaluation = dict(interval=1000)

find_unused_parameters = True

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=0.001,
    step=500,
    gamma=0.97)
runner = dict(type="IterBasedRunner", max_iters=80000)
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
                project = "SSL_segment_h",
                entity = "1820037839-shanghai-jiao-tong-university",
                name = "zs_seg_abd4",
            ),
        )
    ])