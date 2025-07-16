python examples/SeaS_infer.py \
    --output_dir=outputs/images/cable/cut_outer_insulation  \
    --ref_data_dir=data/mvtec_anomaly_detection/cable/train/good/ \
    --gen_model_path=outputs/checkpoints/cable/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/cable/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks17 sks18 sks19 sks20" \
    --total_infer_num=1000 > outputs/logs/infer/cable_infer4.log 2>&1 