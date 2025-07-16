python examples/SeaS_infer.py \
    --output_dir=outputs/images/metal_nut/flip  \
    --ref_data_dir=data/mvtec_anomaly_detection/metal_nut/train/good/ \
    --gen_model_path=outputs/checkpoints/metal_nut/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/metal_nut/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks9 sks10 sks11 sks12" \
    --total_infer_num=1000 > outputs/logs/infer/metal_nut_infer2.log 2>&1 