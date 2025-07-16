python examples/SeaS_infer.py \
    --output_dir=outputs/images/pill/pill_type  \
    --ref_data_dir=data/mvtec_anomaly_detection/pill/train/good/ \
    --gen_model_path=outputs/checkpoints/pill/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/pill/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks21 sks22 sks23 sks24" \
    --total_infer_num=1000 > outputs/logs/infer/pill_infer5.log 2>&1 