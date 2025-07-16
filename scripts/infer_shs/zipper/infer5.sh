python examples/SeaS_infer.py \
    --output_dir=outputs/images/zipper/split_teeth  \
    --ref_data_dir=data/mvtec_anomaly_detection/zipper/train/good/ \
    --gen_model_path=outputs/checkpoints/zipper/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/zipper/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks21 sks22 sks23 sks24" \
    --total_infer_num=1000 > outputs/logs/infer/zipper_infer5.log 2>&1 