python examples/SeaS_infer.py \
    --output_dir=outputs/images/capsule/poke  \
    --ref_data_dir=data/mvtec_anomaly_detection/capsule/train/good/ \
    --gen_model_path=outputs/checkpoints/capsule/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/capsule/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks9 sks10 sks11 sks12" \
    --total_infer_num=1000 > outputs/logs/infer/capsule_infer2.log 2>&1 