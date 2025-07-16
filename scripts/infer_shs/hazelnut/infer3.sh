python examples/SeaS_infer.py \
    --output_dir=outputs/images/hazelnut/print  \
    --ref_data_dir=data/mvtec_anomaly_detection/hazelnut/train/good/ \
    --gen_model_path=outputs/checkpoints/hazelnut/generation-checkpoint \
    --rmp_model_path=outputs/checkpoints/hazelnut/mask-checkpoint/rmp  \
    --prompt="a ob1 with sks13 sks14 sks15 sks16" \
    --total_infer_num=1000 > outputs/logs/infer/hazelnut_infer3.log 2>&1 