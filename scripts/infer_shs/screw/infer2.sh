python examples/SeaS_infer.py \
    --output_dir=outputs/images/screw/scratch_neck  \
    --ref_data_dir=data/mvtec_anomaly_detection/screw/train/good/ \
    --gen_model_path=outputs/checkpoints/screw/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/screw/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks9 sks10 sks11 sks12" \
    --total_infer_num=1000 > outputs/logs/infer/screw_infer2.log 2>&1 