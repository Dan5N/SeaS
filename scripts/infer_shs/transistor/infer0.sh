python examples/SeaS_infer.py \
    --output_dir=outputs/images/transistor/bent_lead  \
    --ref_data_dir=data/mvtec_anomaly_detection/transistor/train/good/ \
    --gen_model_path=outputs/checkpoints/transistor/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/transistor/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks1 sks2 sks3 sks4" \
    --total_infer_num=1000 > outputs/logs/infer/transistor_infer0.log 2>&1 