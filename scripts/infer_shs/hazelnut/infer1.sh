python examples/SeaS_infer.py \
    --output_dir=outputs/images/hazelnut/cut  \
    --ref_data_dir=data/mvtec_anomaly_detection/hazelnut/train/good/ \
    --gen_model_path=outputs/checkpoints/hazelnut/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/hazelnut/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks5 sks6 sks7 sks8" \
    --total_infer_num=1000 > outputs/logs/infer/hazelnut_infer1.log 2>&1 