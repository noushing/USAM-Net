
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image

# =====================================================
# Paths
# =====================================================

dataset1 = Path("/content/Dataset1")
dataset2 = Path("/content/Dataset2")

output = Path("/content/raw_images")

train_dir = output / "train"
val_dir = output / "val"
test_dir = output / "test"

for d in [train_dir, val_dir, test_dir]:
    d.mkdir(parents=True, exist_ok=True)

# =====================================================
# Merge Dataset 1 + Dataset 2
# =====================================================

all_images = []

for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif"]:
    all_images.extend(list(dataset1.glob(ext)))
    all_images.extend(list(dataset2.glob(ext)))

print("Total images:", len(all_images))

# =====================================================
# Split BEFORE augmentation
# =====================================================

train_files, temp_files = train_test_split(
    all_images,
    test_size=0.30,
    random_state=42
)

val_files, test_files = train_test_split(
    temp_files,
    test_size=0.50,
    random_state=42
)

print("Train:", len(train_files))
print("Val:", len(val_files))
print("Test:", len(test_files))

# =====================================================
# Copy Validation and Test
# =====================================================

for f in val_files:
    shutil.copy(f, val_dir / f.name)

for f in test_files:
    shutil.copy(f, test_dir / f.name)

# =====================================================
# Rotation augmentation ONLY for Train
# =====================================================

angles = [0, 90, 180, 270]

for img_path in train_files:

    img = Image.open(img_path)

    for angle in angles:

        rotated = img.rotate(angle, expand=True)

        new_name = (
            img_path.stem +
            f"_rot{angle}" +
            img_path.suffix
        )

        rotated.save(train_dir / new_name)

print("Done.")