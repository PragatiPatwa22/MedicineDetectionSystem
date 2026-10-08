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

extensions = (".jpg", ".jpeg", ".png")

results = {}

print("\n==============================")
print("TESTING ALL MEDICINE IMAGES")
print("==============================\n")

for medicine in os.listdir(DATASET_PATH):

    medicine_path = os.path.join(
        DATASET_PATH,
        medicine
    )

    if not os.path.isdir(medicine_path):
        continue

    total = 0
    correct = 0

    print(f"\n{medicine}")

    for filename in os.listdir(medicine_path):

        if not filename.lower().endswith(extensions):
            continue

        image_path = os.path.join(
            medicine_path,
            filename
        )

        try:
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

            predicted_index = np.argmax(prediction)

            predicted_name = class_names[predicted_index]

            confidence = prediction[predicted_index] * 100

            total += 1

            if predicted_name.lower() == medicine.lower():
                correct += 1

        except Exception as e:
            print(
                f"Error: {filename} - {e}"
            )

    accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )

    results[medicine] = {
        "correct": correct,
        "total": total,
        "accuracy": accuracy
    }

    print(
        f"Correct: {correct}/{total}"
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

print("\n==============================")
print("FINAL RESULTS")
print("==============================")

total_correct = 0
total_images = 0

for medicine, result in results.items():

    print(
        f"{medicine}: "
        f"{result['correct']}/{result['total']} "
        f"= {result['accuracy']:.2f}%"
    )

    total_correct += result["correct"]
    total_images += result["total"]

overall_accuracy = (
    total_correct / total_images * 100
    if total_images > 0
    else 0
)

print("\n==============================")
print(f"Overall Accuracy: {overall_accuracy:.2f}%")
print("==============================")