python examples/SeaS_infer.py \
    --output_dir=outputs/images/screw/thread_top  \
    --ref_data_dir=data/mvtec_anomaly_detection/screw/train/good/ \
    --gen_model_path=outputs/checkpoints/screw/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/screw/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks17 sks18 sks19 sks20" \
    --total_infer_num=1000 > outputs/logs/infer/screw_infer4.log 2>&1 