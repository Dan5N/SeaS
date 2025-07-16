python examples/SeaS_infer.py \
    --output_dir=outputs/images/tile/oil  \
    --ref_data_dir=data/mvtec_anomaly_detection/tile/train/good/ \
    --gen_model_path=outputs/checkpoints/tile/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/tile/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks13 sks14 sks15 sks16" \
    --total_infer_num=1000 > outputs/logs/infer/tile_infer3.log 2>&1 