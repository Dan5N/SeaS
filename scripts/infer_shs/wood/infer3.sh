python examples/SeaS_infer.py \
    --output_dir=outputs/images/wood/liquid  \
    --ref_data_dir=data/mvtec_anomaly_detection/wood/train/good/ \
    --gen_model_path=outputs/checkpoints/wood/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/wood/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks13 sks14 sks15 sks16" \
    --total_infer_num=1000 > outputs/logs/infer/wood_infer3.log 2>&1 