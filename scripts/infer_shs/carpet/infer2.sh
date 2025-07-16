python examples/SeaS_infer.py \
    --output_dir=outputs/images/carpet/hole  \
    --ref_data_dir=data/mvtec_anomaly_detection/carpet/train/good/ \
    --gen_model_path=outputs/checkpoints/carpet/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/carpet/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks9 sks10 sks11 sks12" \
    --total_infer_num=1000 > outputs/logs/infer/carpet_infer2.log 2>&1 