

import cv2
import os
import random
import numpy as np

PROCESSED_DIR = "processed_faces"
MODEL_DIR = "models"
TEST_SPLIT_RATIO = 0.2   # 20% of each player's images held out for testing
RANDOM_SEED = 42         # fixed seed so the split is reproducible every run


def load_dataset():
    """
    Walks through processed_faces/<player_name>/ and builds:
    - image_paths: list of file paths
    - labels: list of integer labels (same length as image_paths)
    - label_to_name: dict mapping integer label -> player name
    """
    image_paths = []
    labels = []
    label_to_name = {}

    player_folders = sorted([
        name for name in os.listdir(PROCESSED_DIR)
        if os.path.isdir(os.path.join(PROCESSED_DIR, name))
    ])

    for label_id, player_name in enumerate(player_folders):
        label_to_name[label_id] = player_name
        player_folder = os.path.join(PROCESSED_DIR, player_name)

        for filename in os.listdir(player_folder):
            if filename.lower().endswith(".jpg"):
                image_paths.append(os.path.join(player_folder, filename))
                labels.append(label_id)

    return image_paths, labels, label_to_name


def train_test_split(image_paths, labels):
   
    random.seed(RANDOM_SEED)

    train_paths, train_labels = [], []
    test_paths, test_labels = [], []

    # Group images by their label first
    by_label = {}
    for path, label in zip(image_paths, labels):
        by_label.setdefault(label, []).append(path)

    for label, paths in by_label.items():
        paths = paths.copy()
        random.shuffle(paths)

        n_test = max(1, round(len(paths) * TEST_SPLIT_RATIO))
        test_set = paths[:n_test]
        train_set = paths[n_test:]

        for p in train_set:
            train_paths.append(p)
            train_labels.append(label)
        for p in test_set:
            test_paths.append(p)
            test_labels.append(label)

    return train_paths, train_labels, test_paths, test_labels


def main():
    if not os.path.isdir(PROCESSED_DIR):
        print(f"ERROR: '{PROCESSED_DIR}' not found. Run preprocess.py first.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    image_paths, labels, label_to_name = load_dataset()

    if len(image_paths) == 0:
        print("ERROR: No processed faces found. Run preprocess.py first.")
        return

    print(f"Loaded {len(image_paths)} total processed faces across "
          f"{len(label_to_name)} players.\n")

    train_paths, train_labels, test_paths, test_labels = train_test_split(
        image_paths, labels
    )

    print(f"Train set: {len(train_paths)} images")
    print(f"Test set:  {len(test_paths)} images\n")

    # Load actual pixel data for training images
    train_images = []
    for path in train_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        train_images.append(img)

    # Create and train the LBPH recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(train_images, np.array(train_labels))

    # Save the trained model
    model_path = os.path.join(MODEL_DIR, "lbph_model.yml")
    recognizer.save(model_path)
    print(f"Model saved to: {model_path}")

    # Save label -> name mapping
    labels_path = os.path.join(MODEL_DIR, "labels.txt")
    with open(labels_path, "w") as f:
        for label_id, name in label_to_name.items():
            f.write(f"{label_id},{name}\n")
    print(f"Label mapping saved to: {labels_path}")

    # Save the test set file list so evaluate.py uses the SAME split
    test_split_path = os.path.join(MODEL_DIR, "test_split.txt")
    with open(test_split_path, "w") as f:
        for path, label in zip(test_paths, test_labels):
            f.write(f"{path},{label}\n")
    print(f"Test split saved to: {test_split_path}")

    print("\nTraining complete. Run evaluate.py next to check accuracy.")


if __name__ == "__main__":
    main()