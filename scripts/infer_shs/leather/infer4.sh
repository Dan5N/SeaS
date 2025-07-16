python examples/SeaS_infer.py \
    --output_dir=outputs/images/leather/poke  \
    --ref_data_dir=data/mvtec_anomaly_detection/leather/train/good/ \
    --gen_model_path=outputs/checkpoints/leather/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/leather/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks17 sks18 sks19 sks20" \
    --total_infer_num=1000 > outputs/logs/infer/leather_infer4.log 2>&1 