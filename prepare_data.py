import os
import shutil
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------------- paths ----------------
dataset_dir = r"/home/doaa/DL/Data"
output_dir = "processed_dataset"     # train/val/test

# ---------------- استخراج كل الأشخاص ----------------
selected_people = [p for p in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, p))]
print("Number of classes:", len(selected_people))

# ---------------- make folders Train/Val/Test ----------------
splits = ["train", "val", "test"]
for split in splits:
    for person in selected_people:
        os.makedirs(os.path.join(output_dir, split, person), exist_ok=True)

# ---------------- split image----------------
for person in selected_people:
    person_path = os.path.join(dataset_dir, person)
    images = os.listdir(person_path)

    random.shuffle(images)

    n = len(images)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    split_images = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, imgs in split_images.items():
        for img in imgs:
            src = os.path.join(person_path, img)
            dst = os.path.join(output_dir, split, person, img)
            shutil.copy2(src, dst)

print("✅ Dataset split completed successfully!")

# ---------------- Augmentation و Transforms ----------------
img_size = 160  # Resize to image 

# Augmentation 
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

#  resize + normalize
test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ---------------- Load Datasets ----------------
train_data = datasets.ImageFolder(os.path.join(output_dir, "train"), transform=train_transform)
val_data   = datasets.ImageFolder(os.path.join(output_dir, "val"), transform=test_transform)
test_data  = datasets.ImageFolder(os.path.join(output_dir, "test"), transform=test_transform)

# ---------------- DataLoaders ----------------
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=8)
test_loader  = DataLoader(test_data, batch_size=8)

print("Classes:", train_data.classes)
print("Number of training images:", len(train_data))
print("Number of validation images:", len(val_data))
print("Number of test images:", len(test_data))
