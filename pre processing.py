# =========================
# IMPORT LIBRARIES
# =========================
import os
import cv2
import numpy as np
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================
IMG_SIZE = 224  # Required input size for VGG16/ResNet50
DATA_DIR = "dataset"  # root folder

CLASSES = ["benign", "malignant"]

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess_image(img_path):
    # Read image
    img = cv2.imread(img_path)

    # Resize to 224x224
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize (0–255 → 0–1)
    img = img / 255.0

    # Contrast Enhancement (optional but good)
    img = cv2.equalizeHist((img * 255).astype(np.uint8)) / 255.0

    # Expand dimensions (needed for CNN input)
    img = np.expand_dims(img, axis=-1)  # (224,224) → (224,224,1)

    return img

# =========================
# LOAD DATASET
# =========================
data = []
labels = []

for label, category in enumerate(CLASSES):
    path = os.path.join(DATA_DIR, category)

    for img_name in tqdm(os.listdir(path)):
        try:
            img_path = os.path.join(path, img_name)

            img = preprocess_image(img_path)

            data.append(img)
            labels.append(label)

        except Exception as e:
            continue

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)
