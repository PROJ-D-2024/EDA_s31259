import hashlib
import os
import shutil
import csv

DATASETS = [
    "datasets_orginal/dataset1/human-cctv-.v1i.yolov8",
    "datasets_orginal/dataset2/Sourced-Human-Detection-CCTV.v2-roboflow-instant-1--eval-.yolov8",
    "datasets_orginal/dataset3/Human CCTV.v7i.yolov8"
]

OUTPUT = "dataset_eda"
IMAGES = os.path.join(OUTPUT, "images")
LABELS = os.path.join(OUTPUT, "labels.csv")

os.makedirs(IMAGES, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)

csv_rows = []
image_counter = 1

def image_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

hashes = {}
duplicates = 0

for dataset_path in DATASETS:
    for split in ["train", "valid", "test"]:
        images_dir = os.path.join(dataset_path, split, "images")
        labels_dir = os.path.join(dataset_path, split, "labels")

        for img_file in os.listdir(images_dir):

            src_img = os.path.join(images_dir, img_file)
            img_hash = image_hash(src_img)

            if img_hash in hashes:
                duplicates += 1
                continue

            image_id = image_counter
            new_image_name = f"image_{image_counter:05d}.jpg"

            hashes[img_hash] = new_image_name

            dst_img = os.path.join(IMAGES, new_image_name)
            shutil.copy(src_img, dst_img)

            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_file)

            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()

                        if len(parts) < 5:
                            continue

                        class_id, x, y, w, h = parts[:5]

                        csv_rows.append([
                            image_id,
                            int(class_id),
                            float(x),
                            float(y),
                            float(w),
                            float(h)
                        ])

            image_counter += 1


with open(LABELS, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_id",
        "class",
        "x_center",
        "y_center",
        "width",
        "height"
    ])
    writer.writerows(csv_rows)

print(duplicates)