python examples/SeaS_infer.py \
    --output_dir=outputs/images/zipper/rough  \
    --ref_data_dir=data/mvtec_anomaly_detection/zipper/train/good/ \
    --gen_model_path=outputs/checkpoints/zipper/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/zipper/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks17 sks18 sks19 sks20" \
    --total_infer_num=1000 > outputs/logs/infer/zipper_infer4.log 2>&1 