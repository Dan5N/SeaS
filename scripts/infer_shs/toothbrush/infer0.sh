python examples/SeaS_infer.py \
    --output_dir=outputs/images/toothbrush/defective  \
    --ref_data_dir=data/mvtec_anomaly_detection/toothbrush/train/good/ \
    --gen_model_path=outputs/checkpoints/toothbrush/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/toothbrush/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks1 sks2 sks3 sks4" \
    --total_infer_num=1000 > outputs/logs/infer/toothbrush_infer0.log 2>&1 