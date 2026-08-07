import os
import shutil
import random

# Source path where your images and label .txt files are located
SOURCE_DIR = r"C:\Users\admin\Desktop\ka_data_ppe\dataset\images"

# Target base path for organized dataset
DATASET_DIR = r"C:\Users\admin\Desktop\ka_data_ppe\dataset"

# Target directories
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LBL_DIR = os.path.join(DATASET_DIR, "train", "labels")
VAL_IMG_DIR   = os.path.join(DATASET_DIR, "val", "images")
VAL_LBL_DIR   = os.path.join(DATASET_DIR, "val", "labels")

# Create directories if they don't exist
for folder in [TRAIN_IMG_DIR, TRAIN_LBL_DIR, VAL_IMG_DIR, VAL_LBL_DIR]:
    os.makedirs(folder, exist_ok=True)

# Supported image extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

# Find all image files
all_files = os.listdir(SOURCE_DIR)
image_files = [f for f in all_files if f.lower().endswith(IMAGE_EXTENSIONS)]

# Shuffle for random train/val split (80% train, 20% validation)
random.seed(42)
random.shuffle(image_files)

split_idx = int(len(image_files) * 0.8)
train_images = image_files[:split_idx]
val_images = image_files[split_idx:]

def move_pair(img_file, target_img_dir, target_lbl_dir):
    base_name = os.path.splitext(img_file)[0]
    txt_file = base_name + ".txt"

    src_img_path = os.path.join(SOURCE_DIR, img_file)
    src_lbl_path = os.path.join(SOURCE_DIR, txt_file)

    if os.path.exists(src_img_path):
        shutil.copy(src_img_path, os.path.join(target_img_dir, img_file))

    if os.path.exists(src_lbl_path):
        shutil.copy(src_lbl_path, os.path.join(target_lbl_dir, txt_file))

# Copy training files
for img in train_images:
    move_pair(img, TRAIN_IMG_DIR, TRAIN_LBL_DIR)

# Copy validation files
for img in val_images:
    move_pair(img, VAL_IMG_DIR, VAL_LBL_DIR)

print("\n--- DATASET RESTRUCTURED SUCCESSFULLY ---")
print(f"Total Images Found : {len(image_files)}")
print(f"Training Set (80%) : {len(train_images)} images -> dataset/train/")
print(f"Validation Set (20%): {len(val_images)} images -> dataset/val/")
print("------------------------------------------\n")