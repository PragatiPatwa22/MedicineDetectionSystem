import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "medicine_model.keras"
CLASS_FILE = "class_indices.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_FILE, "r") as file:
    class_indices = json.load(file)

class_names = [None] * len(class_indices)

for name, index in class_indices.items():
    class_names[int(index)] = name

image_path = input("Enter image path: ")

image = Image.open(image_path).convert("RGB")
image = image.resize((224, 224))

image_array = np.array(image).astype(np.float32)
image_array = preprocess_input(image_array)
image_array = np.expand_dims(image_array, axis=0)

prediction = model.predict(image_array, verbose=0)[0]

results = []

for i, probability in enumerate(prediction):
    results.append((class_names[i], float(probability) * 100))

results.sort(key=lambda x: x[1], reverse=True)

print("\nPrediction Results:")
print("-------------------")

for medicine, confidence in results:
    print(f"{medicine}: {confidence:.2f}%")

print("\nTop Prediction:")
print(results[0][0])
print(f"Confidence: {results[0][1]:.2f}%")