import os
from PIL import Image

DATASET_PATH = "medicine_images"

print("\nIMAGE SIZE CHECK")
print("============================")

for medicine in sorted(os.listdir(DATASET_PATH)):

    folder = os.path.join(DATASET_PATH, medicine)

    if not os.path.isdir(folder):
        continue

    print(f"\n{medicine}")
    print("----------------------------")

    widths = []
    heights = []

    for filename in os.listdir(folder):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        path = os.path.join(folder, filename)

        try:
            with Image.open(path) as image:
                width, height = image.size

            widths.append(width)
            heights.append(height)

        except:
            pass

    if len(widths) > 0:

        print(f"Images: {len(widths)}")
        print(f"Smallest width: {min(widths)}")
        print(f"Largest width: {max(widths)}")
        print(f"Smallest height: {min(heights)}")
        print(f"Largest height: {max(heights)}")
        print(
            f"Average width: "
            f"{sum(widths) / len(widths):.0f}"
        )
        print(
            f"Average height: "
            f"{sum(heights) / len(heights):.0f}"
        )

print("\n============================")
print("Image size check completed.")
print("============================")