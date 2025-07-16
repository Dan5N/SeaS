python examples/SeaS_infer.py \
    --output_dir=outputs/images/grid/metal_contamination  \
    --ref_data_dir=data/mvtec_anomaly_detection/grid/train/good/ \
    --gen_model_path=outputs/checkpoints/grid/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/grid/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks13 sks14 sks15 sks16" \
    --total_infer_num=1000 > outputs/logs/infer/grid_infer3.log 2>&1 