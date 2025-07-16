python examples/SeaS_infer.py \
    --output_dir=outputs/images/wood/hole  \
    --ref_data_dir=data/mvtec_anomaly_detection/wood/train/good/ \
    --gen_model_path=outputs/checkpoints/wood/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/wood/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks9 sks10 sks11 sks12" \
    --total_infer_num=1000 > outputs/logs/infer/wood_infer2.log 2>&1 