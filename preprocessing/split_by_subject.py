import os
import re
import shutil
import random
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

def split_dataset_by_subject(
    raw_data_dir,
    metadata_excel_path,
    output_root,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42
):
    # Step 1: Extract subject IDs from image filenames
    subject_image_map = {}
    for class_name in os.listdir(raw_data_dir):
        class_path = os.path.join(raw_data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            if not fname.endswith(".png"):
                continue
            match = re.match(r"(OAS\d{1}_\d{4}_MR\d_\d)", fname)
            if match:
                subject_id = match.group(1)
                full_path = os.path.join(class_path, fname)
                subject_image_map.setdefault(subject_id, []).append((full_path, class_name))

    # Step 2: Shuffle and split subjects
    random.seed(seed)
    all_subjects = list(subject_image_map.keys())
    random.shuffle(all_subjects)

    n_total = len(all_subjects)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_ids = set(all_subjects[:n_train])
    val_ids = set(all_subjects[n_train:n_train + n_val])
    test_ids = set(all_subjects[n_train + n_val:])

    # Step 3: Organize images by split
    split_to_images = defaultdict(list)
    for subject_id, files in subject_image_map.items():
        if subject_id in train_ids:
            split_to_images["train"].extend(files)
        elif subject_id in val_ids:
            split_to_images["val"].extend(files)
        elif subject_id in test_ids:
            split_to_images["test"].extend(files)

    # Step 4: Create output folders and copy files
    for split in ["train", "val", "test"]:
        for class_name in ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']:
            os.makedirs(os.path.join(output_root, split, class_name), exist_ok=True)

    for split in ["train", "val", "test"]:
        for img_path, class_name in tqdm(split_to_images[split], desc=f"Copying {split}"):
            dest = os.path.join(output_root, split, class_name, os.path.basename(img_path))
            shutil.copy(img_path, dest)

    # Summary
    for split in ["train", "val", "test"]:
        print(f"{split.capitalize()} set: {len(split_to_images[split])} images")

    print("Subject-level dataset split complete.")
