python examples/SeaS_infer.py \
    --output_dir=outputs/images/pill/scratch  \
    --ref_data_dir=data/mvtec_anomaly_detection/pill/train/good/ \
    --gen_model_path=outputs/checkpoints/pill/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/pill/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks25 sks26 sks27 sks28" \
    --total_infer_num=1000 > outputs/logs/infer/pill_infer6.log 2>&1 