import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

MODEL_PATH = "medicine_model.keras"
DATASET_PATH = "prepared_images"

model = tf.keras.models.load_model(MODEL_PATH)

with open("class_indices.json", "r") as file:
    class_indices = json.load(file)

class_names = [None] * len(class_indices)

for name, index in class_indices.items():
    class_names[int(index)] = name

paracetamol_path = os.path.join(
    DATASET_PATH,
    "paracetamol"
)

extensions = (".jpg", ".jpeg", ".png")

files = [
    filename
    for filename in os.listdir(paracetamol_path)
    if filename.lower().endswith(extensions)
]

files.sort()

print("\n==============================")
print("PARACETAMOL IMAGE TEST")
print("==============================")

print(f"\nTotal Paracetamol images found: {len(files)}")

correct = 0

for filename in files:

    image_path = os.path.join(
        paracetamol_path,
        filename
    )

    img = image.load_img(
        image_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_array
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )[0]

    top_indices = np.argsort(
        prediction
    )[::-1][:3]

    predicted_index = top_indices[0]

    if class_names[predicted_index].lower() == "paracetamol":
        correct += 1

    print(f"\n{filename}")

    for index in top_indices:
        print(
            f"{class_names[index]}: "
            f"{prediction[index] * 100:.2f}%"
        )

print("\n==============================")
print("PARACETAMOL SUMMARY")
print("==============================")

print(
    f"Correct: {correct}/{len(files)}"
)

if len(files) > 0:
    accuracy = correct / len(files) * 100
    print(
        f"Accuracy: {accuracy:.2f}%"
    )

print("==============================")