python examples/SeaS_infer.py \
    --output_dir=outputs/images/grid/thread  \
    --ref_data_dir=data/mvtec_anomaly_detection/grid/train/good/ \
    --gen_model_path=outputs/checkpoints/grid/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/grid/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks17 sks18 sks19 sks20" \
    --total_infer_num=1000 > outputs/logs/infer/grid_infer4.log 2>&1 