python examples/SeaS_infer.py \
    --output_dir=outputs/images/metal_nut/scratch  \
    --ref_data_dir=data/mvtec_anomaly_detection/metal_nut/train/good/ \
    --gen_model_path=outputs/checkpoints/metal_nut/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/metal_nut/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks13 sks14 sks15 sks16" \
    --total_infer_num=1000 > outputs/logs/infer/metal_nut_infer3.log 2>&1 