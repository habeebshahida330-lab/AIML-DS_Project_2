"""
evaluate.py - Step 5: Test the model on unseen images and measure real accuracy.
"""

import cv2
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

MODEL_DIR = "models"

def load_label_map():
    label_to_name = {}
    with open(os.path.join(MODEL_DIR, "labels.txt")) as f:
        for line in f:
            label_id, name = line.strip().split(",")
            label_to_name[int(label_id)] = name
    return label_to_name

def load_test_set():
    paths, labels = [], []
    with open(os.path.join(MODEL_DIR, "test_split.txt")) as f:
        for line in f:
            path, label = line.strip().rsplit(",", 1)
            paths.append(path)
            labels.append(int(label))
    return paths, labels

def main():
    label_to_name = load_label_map()
    test_paths, true_labels = load_test_set()

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join(MODEL_DIR, "lbph_model.yml"))

    predicted_labels = []
    print("Predictions on test set:\n")
    for path, true_label in zip(test_paths, true_labels):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        pred_label, confidence = recognizer.predict(img)
        predicted_labels.append(pred_label)
        true_name = label_to_name[true_label]
        pred_name = label_to_name[pred_label]
        correct = "OK" if pred_label == true_label else "WRONG"
        print(f"[{correct}] True: {true_name:20s} Predicted: {pred_name:20s} "
              f"(distance={confidence:.1f}) file={os.path.basename(path)}")

    acc = accuracy_score(true_labels, predicted_labels)
    print(f"\nOverall Test Accuracy: {acc*100:.2f}%")
    print("(Note: LBPH 'confidence' is actually a DISTANCE score - LOWER means more confident/similar.)")

    names = [label_to_name[i] for i in sorted(label_to_name)]
    cm = confusion_matrix(true_labels, predicted_labels, labels=sorted(label_to_name))

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix (Accuracy: {acc*100:.1f}%)")
    plt.colorbar()
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.yticks(range(len(names)), names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(len(names)):
        for j in range(len(names)):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
    print(f"\nConfusion matrix saved to: {MODEL_DIR}/confusion_matrix.png")

if __name__ == "__main__":
    main()