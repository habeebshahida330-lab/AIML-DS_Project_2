import cv2

# Load the trained recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/lbph_model.yml")

# Load label map (format: "0,kl_rahul")
label_to_name = {}
with open("models/labels.txt", "r") as f:
    for line in f:
        label_id, name = line.strip().split(",")
        label_to_name[int(label_id)] = name

print(f"{'Actual Player':<20} {'Predicted':<20} {'Distance':<10} {'Correct?'}")
print("-" * 65)

# Read test_split.txt (format: "path,label_id")
with open("models/test_split.txt", "r") as f:
    for line in f:
        img_path, true_label = line.strip().rsplit(",", 1)
        true_label = int(true_label)
        actual_player = label_to_name[true_label]

        face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if face_img is None:
            print(f"{actual_player:<20} {'IMAGE NOT FOUND':<20}")
            continue

        face_img = cv2.resize(face_img, (200, 200))

        predicted_label, distance = recognizer.predict(face_img)
        predicted_player = label_to_name[predicted_label]

        correct = "YES" if predicted_player == actual_player else "NO"
        print(f"{actual_player:<20} {predicted_player:<20} {distance:<10.2f} {correct}")