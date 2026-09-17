"""
preprocess.py
--------------
Step 3 of the Indian Cricket Team Member Face Identification project.

WHAT THIS SCRIPT DOES (in plain English):
1. Goes through every player folder inside dataset/
2. Opens each image
3. Uses OpenCV's Haar Cascade classifier to find a face in the image
4. Crops just the face region
5. Converts it to grayscale (LBPH works on grayscale intensity patterns, not color)
6. Resizes every face to the SAME fixed size (200x200) so all images are comparable
7. Applies histogram equalization (evens out lighting differences between photos)
8. Saves the cleaned-up face into processed_faces/<player_name>/

WHY THESE STEPS:
- Haar Cascade: a classical, pre-trained detector built into OpenCV. It looks for
  patterns of light/dark rectangles typical of a face (eyes are darker than cheeks, etc.)
  It does not need any training from us - it ships ready-to-use with OpenCV.
- Grayscale: LBPH (used later in train.py) is a texture-based algorithm that works on
  pixel intensity patterns. Color adds no value here and only adds noise.
- Fixed size (200x200): LBPH compares histograms between faces. If images are different
  sizes, the histograms will not be comparable. Resizing standardizes everything.
- Histogram equalization: photos are taken in very different lighting conditions
  (stadium lights, indoor interviews, sunny outdoor shots). Equalization stretches
  the contrast so the model focuses on facial structure, not brightness differences.
"""

import cv2
import os

# ---------------------------------------------------------------------------
# CONFIGURATION - change these if your folder names or sizes differ
# ---------------------------------------------------------------------------
DATASET_DIR = "dataset"                # where your raw player photos are
OUTPUT_DIR = "processed_faces"         # where cleaned face crops will be saved
FACE_SIZE = (200, 200)                 # every saved face will be resized to this

# Load OpenCV's built-in pre-trained Haar Cascade face detector.
# This XML file ships with opencv-python - no need to download it separately.
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(CASCADE_PATH)


def process_player_folder(player_name):
    """
    Processes every image inside dataset/<player_name>/
    Detects a face, crops it, cleans it up, and saves it into
    processed_faces/<player_name>/
    Returns (success_count, fail_count) for reporting.
    """
    input_folder = os.path.join(DATASET_DIR, player_name)
    output_folder = os.path.join(OUTPUT_DIR, player_name)
    os.makedirs(output_folder, exist_ok=True)

    success_count = 0
    fail_count = 0

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Skip anything that is not a normal image file
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            print(f"  [SKIPPED - unsupported format] {filename}")
            continue

        # Read the image from disk
        image = cv2.imread(file_path)
        if image is None:
            print(f"  [FAILED TO READ] {filename} (file may be corrupted)")
            fail_count += 1
            continue

        # Convert to grayscale BEFORE detection - Haar Cascade works on grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image.
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            print(f"  [NO FACE DETECTED] {filename}")
            fail_count += 1
            continue

        # If multiple faces are detected in one photo, take the LARGEST one.
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]

        # Crop just the face region out of the grayscale image
        face_crop = gray[y:y + h, x:x + w]

        # Resize to a fixed standard size
        face_resized = cv2.resize(face_crop, FACE_SIZE)

        # Histogram equalization - normalizes lighting/contrast differences
        face_equalized = cv2.equalizeHist(face_resized)

        # Save the processed face
        out_name = f"{os.path.splitext(filename)[0]}_face.jpg"
        out_path = os.path.join(output_folder, out_name)
        cv2.imwrite(out_path, face_equalized)

        success_count += 1

    return success_count, fail_count


def main():
    if not os.path.isdir(DATASET_DIR):
        print(f"ERROR: '{DATASET_DIR}' folder not found. "
              f"Make sure you're running this from the project root.")
        return

    player_folders = [
        name for name in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, name))
    ]

    if not player_folders:
        print(f"ERROR: No player folders found inside '{DATASET_DIR}'.")
        return

    print(f"Found {len(player_folders)} player folders: {player_folders}\n")

    total_success = 0
    total_fail = 0

    for player in player_folders:
        print(f"Processing '{player}'...")
        success, fail = process_player_folder(player)
        print(f"  -> {success} face(s) saved, {fail} image(s) failed/skipped\n")
        total_success += success
        total_fail += fail

    print("=" * 50)
    print(f"DONE. Total faces saved: {total_success}")
    print(f"Total failed/skipped:    {total_fail}")
    print(f"Processed faces are in:  '{OUTPUT_DIR}/'")
    print("=" * 50)

    if total_success < 20:
        print("\nWARNING: Very few faces were successfully processed.")
        print("The model trained on this data will likely have low accuracy.")
        print("Consider adding more, clearer, front-facing photos per player.")


if __name__ == "__main__":
    main()