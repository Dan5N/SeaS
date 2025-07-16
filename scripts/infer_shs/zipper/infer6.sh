python examples/SeaS_infer.py \
    --output_dir=outputs/images/zipper/squeezed_teeth  \
    --ref_data_dir=data/mvtec_anomaly_detection/zipper/train/good/ \
    --gen_model_path=outputs/checkpoints/zipper/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/zipper/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks25 sks26 sks27 sks28" \
    --total_infer_num=1000 > outputs/logs/infer/zipper_infer6.log 2>&1 