python step0_preprocess_totalsegmentator.py --max_workers 8 \
    --totalsegmentator_path_v1 "/path/to/Data/Totalsegmentator_dataset_v1/" \
    --totalsegmentator_path_v2 "/path/to/Data/Totalsegmentator_dataset_v201/" \
    --savedir "/path/to/Data_gen/Totalsegmentator_gen_seg/"
CUDA_VISIBLE_DEVICES=0 python step1_generate_label_maps.py --max_workers 8 \
    --savedir "/path/to/Data_gen/Totalsegmentator_gen_seg/"
CUDA_VISIBLE_DEVICES=0 python step2_generate_views_cuda.py --max_workers 8 \
    --savedir "/path/to/Data_gen/Totalsegmentator_gen_seg/"
python step3_gen_abd4_labels.py --data_dir "/path/to/Data_gen/Totalsegmentator_gen_seg/"