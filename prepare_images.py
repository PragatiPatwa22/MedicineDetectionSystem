import os
from PIL import Image, ImageOps

SOURCE_FOLDER = "medicine_images"
OUTPUT_FOLDER = "prepared_images"
IMAGE_SIZE = (224, 224)

extensions = (".jpg", ".jpeg", ".png")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("\nPreparing medicine images...\n")

total = 0

for medicine in os.listdir(SOURCE_FOLDER):

    source_path = os.path.join(
        SOURCE_FOLDER,
        medicine
    )

    if not os.path.isdir(source_path):
        continue

    output_path = os.path.join(
        OUTPUT_FOLDER,
        medicine
    )

    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(source_path):

        if not filename.lower().endswith(extensions):
            continue

        input_file = os.path.join(
            source_path,
            filename
        )

        output_file = os.path.join(
            output_path,
            filename
        )

        try:
            image = Image.open(
                input_file
            ).convert("RGB")

            image = ImageOps.fit(
                image,
                IMAGE_SIZE,
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5)
            )

            image.save(
                output_file,
                quality=95
            )

            total += 1

            print(
                f"Processed: {medicine}/{filename}"
            )

        except Exception as e:
            print(
                f"Error: {input_file} - {e}"
            )

print("\n============================")
print(f"Total images prepared: {total}")
print("============================")