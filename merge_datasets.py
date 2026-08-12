import os
import shutil

source_folders = [
    r"dataset\V0",
    r"dataset\V1",
    r"dataset\V2",
    r"dataset\V3",
    r"dataset\V4",
]

dest_images = r"dataset\train\images"
dest_labels = r"dataset\train\labels"

os.makedirs(dest_images, exist_ok=True)
os.makedirs(dest_labels, exist_ok=True)

total_moved = 0

for folder in source_folders:
  if not os.path.exists(folder):
    print(f"Skipping {folder} (does not exist)")
    continue

  print(f"Scanning {folder}...")

  # Search recursively or check common subfolder patterns
  found_files = False
  for root, dirs, files in os.walk(folder):
    # Look for image files in current walk directory
    img_files = [
        f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if img_files:
      print(f" -> Found {len(img_files)} images in {os.path.relpath(root, '.')}")
      prefix = os.path.basename(folder)

      for img_file in img_files:
        src_img = os.path.join(root, img_file)
        new_img_name = f"{prefix}_{img_file}"
        dst_img = os.path.join(dest_images, new_img_name)
        shutil.copy(src_img, dst_img)

        # Look for corresponding .txt label file in the same directory or a parallel 'labels' folder
        base_name, _ = os.path.splitext(img_file)
        lbl_file = base_name + ".txt"

        src_lbl = os.path.join(root, lbl_file)
        if not os.path.exists(src_lbl):
          # Try checking if there's a sibling 'labels' folder
          parent_dir = os.path.dirname(root)
          alt_lbl_dir = os.path.join(parent_dir, "labels")
          src_lbl = os.path.join(alt_lbl_dir, lbl_file)

        if os.path.exists(src_lbl):
          new_lbl_name = f"{prefix}_{lbl_file}"
          dst_lbl = os.path.join(dest_labels, new_lbl_name)
          shutil.copy(src_lbl, dst_lbl)
          total_moved += 1

      found_files = True

  if not found_files:
    print(f" -> No images found in {folder}")

print(f"\nSuccessfully merged files! Total training samples ready: {total_moved}")