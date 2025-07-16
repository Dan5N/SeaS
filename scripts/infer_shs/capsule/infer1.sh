python examples/SeaS_infer.py \
    --output_dir=outputs/images/capsule/faulty_imprint  \
    --ref_data_dir=data/mvtec_anomaly_detection/capsule/train/good/ \
    --gen_model_path=outputs/checkpoints/capsule/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/capsule/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks5 sks6 sks7 sks8" \
    --total_infer_num=1000 > outputs/logs/infer/capsule_infer1.log 2>&1 