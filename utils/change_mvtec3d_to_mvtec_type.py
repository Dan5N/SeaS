#!/usr/bin/env python3
import os
import shutil

def reorganize_structure_copy(root_dir, target_dir):
    """
    this function change the MVTec-3D-AD original structure:
        product/
        ├── test/
        │   └── anomlay_type/
        │       ├── rgb/
        │       │   ├── a.png
        │       │   └── b.png
        │       └── gt/
        │           ├── a.png
        │           └── b.png
        └── train/
            └── good/
                └── rgb/
                    ├── a.png
                    └── b.png

    to a MVTec-AD-like structure:
        product/
        ├── test/
        │   └── anomlay_type/
        │       ├── a.png
        │       └── b.png
        ├── ground_truth/
        │   └── anomlay_type/
        │       ├── a.png
        │       └── b.png
        └── train/
            └── good/
                ├── a.png
                └── b.png
    """
    test_src    = os.path.join(root_dir, "test")
    train_good  = os.path.join(root_dir, "train", "good")

    # 0. loop over anomaly type (short for anotype)
    for anotype in os.listdir(test_src):
        anotype_src = os.path.join(test_src, anotype)
        if not os.path.isdir(anotype_src): continue

        # 1. copy rgb -> target_dir/test/<anotype>
        rgb_src = os.path.join(anotype_src, "rgb")
        dst_test_anotype = os.path.join(target_dir, "test", anotype)
        if os.path.isdir(rgb_src):
            os.makedirs(dst_test_anotype, exist_ok=True)
            for fname in os.listdir(rgb_src):
                shutil.copy2(os.path.join(rgb_src, fname),
                             os.path.join(dst_test_anotype, fname))

        # 2. copy gt  -> target_dir/ground_truth/<anotype>
        gt_src = os.path.join(anotype_src, "gt")
        dst_gt_anotype = os.path.join(target_dir, "ground_truth", anotype)
        if os.path.isdir(gt_src):
            os.makedirs(dst_gt_anotype, exist_ok=True)
            for fname in os.listdir(gt_src):
                shutil.copy2(os.path.join(gt_src, fname),
                             os.path.join(dst_gt_anotype, fname))

    # 3. copy train/good/rgb -> target_dir/train/good
    rgb_train = os.path.join(train_good, "rgb")
    dst_train  = os.path.join(target_dir, "train", "good")
    if os.path.isdir(rgb_train):
        os.makedirs(dst_train, exist_ok=True)
        for fname in os.listdir(rgb_train):
            shutil.copy2(os.path.join(rgb_train, fname),
                         os.path.join(dst_train, fname))

if __name__ == "__main__":
    products = ["bagel"
                "cable_gland"
                "carrot"
                "cookie"
                "dowel"
                "foam"
                "peach"
                "potato"
                "rope"
                "tire"]
    for product in products:
        src_root = f"data/mvtec_3d_anomaly_detection/{product}"
        dst_root = f"data/mvtec_3d_anomaly_detection_reorg/{product}"
        reorganize_structure_copy(src_root, dst_root)
    print("Reorganization with copy completed.")

