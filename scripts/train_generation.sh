# MVTec AD
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_FORCE_NUM_PROCESSES=1

declare -A subclasses_nums=(
  [bottle]=3     [cable]=8      [capsule]=5   [carpet]=5
  [grid]=5       [hazelnut]=4   [leather]=5   [metal_nut]=4
  [pill]=7       [screw]=5      [tile]=5      [toothbrush]=1
  [transistor]=4 [wood]=5       [zipper]=7
)

for category in $(printf "%s\n" "${!subclasses_nums[@]}" | sort); do
  echo "==== Processing train generation for $category ===="
  export INSTANCE_DIR="data/mvtec_anomaly_detection/${category}/test/"
  export MASK_DIR="data/mvtec_anomaly_detection/${category}/ground_truth/"
  export NORMAL_DIR="data/mvtec_anomaly_detection/${category}/train/good/"
  export OUTPUT_DIR="outputs/checkpoints/${category}"

  gen_steps=$(( subclasses_nums[$category] * 800 ))

  accelerate launch --config_file configs/accelerater_config.yaml examples/SeaS_main.py \
    --output_dir=$OUTPUT_DIR \
    --instance_data_dir=$INSTANCE_DIR \
    --mask_dir=$MASK_DIR \
    --normal_data_dir=$NORMAL_DIR \
    --gen_train_steps=$gen_steps \
    --checkpointing_steps=$gen_steps > outputs/logs/train_generation/${category}.log 2>&1

  echo "  ---- Done train generation for $category ----"
done


# # MVTec 3D AD
# #!/usr/bin/env bash
# export CUDA_VISIBLE_DEVICES=0
# export ACCELERATE_FORCE_NUM_PROCESSES=1

# declare -A subclasses_nums=(
#   [bagel]=4    [cable_gland]=4    [carrot]=5
#   [cookie]=4   [dowel]=4          [foam]=4
#   [peach]=4    [potato]=4         [rope]=3
#   [tire]=4
# )

# for category in $(printf "%s\n" "${!subclasses_nums[@]}" | sort); do
#   echo "==== Processing train generation for $category ===="
#   export INSTANCE_DIR="data/mvtec_3d_anomaly_detection_reorg/${category}/test/"
#   export MASK_DIR="data/mvtec_3d_anomaly_detection_reorg/${category}/ground_truth/"
#   export NORMAL_DIR="data/mvtec_3d_anomaly_detection_reorg/${category}/train/good/"
#   export OUTPUT_DIR="outputs/checkpoints/${category}"

#   gen_steps=$(( subclasses_nums[$category] * 800 ))

#   accelerate launch --config_file configs/accelerater_config.yaml examples/SeaS_main.py \
#     --output_dir=$OUTPUT_DIR \
#     --instance_data_dir=$INSTANCE_DIR \
#     --mask_dir=$MASK_DIR \
#     --normal_data_dir=$NORMAL_DIR \
#     --gen_train_steps=$gen_steps \
#     --checkpointing_steps=$gen_steps > outputs/logs/train_generation/${category}.log 2>&1
#
#   echo "  ---- Done train generation for $category ----"
# done


# # VisA
# #!/usr/bin/env bash
# export CUDA_VISIBLE_DEVICES=0
# export ACCELERATE_FORCE_NUM_PROCESSES=1

# declare -A subclasses_nums=(
#   [candle]=8       [capsules]=2     [cashew]=9       [chewinggum]=5
#   [fryum]=8        [macaroni1]=6    [macaroni2]=7    [pcb1]=5
#   [pcb2]=5         [pcb3]=5         [pcb4]=7         [pipe_fryum]=7
# )

# for category in $(printf "%s\n" "${!subclasses_nums[@]}" | sort); do
#   echo "==== Processing train generation for $category ===="
#   export INSTANCE_DIR="data/visa/${category}/test/"
#   export MASK_DIR="data/visa/${category}/ground_truth/"
#   export NORMAL_DIR="data/visa/${category}/train/good/"
#   export OUTPUT_DIR="outputs/checkpoints/${category}"

#   gen_steps=$(( subclasses_nums[$category] * 800 ))

#   accelerate launch --config_file configs/accelerater_config.yaml examples/SeaS_main.py \
#     --output_dir=$OUTPUT_DIR \
#     --instance_data_dir=$INSTANCE_DIR \
#     --mask_dir=$MASK_DIR \
#     --normal_data_dir=$NORMAL_DIR \
#     --gen_train_steps=$gen_steps \
#     --checkpointing_steps=$gen_steps > outputs/logs/train_generation/${category}.log 2>&1

#   echo "  ---- Done train generation for $category ----"
# done

