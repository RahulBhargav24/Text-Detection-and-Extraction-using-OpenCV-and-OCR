import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Step 1: Load Image
# -----------------------------
image_path = r"D:\Projects\test img\testphoto2.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    exit()

# -----------------------------
# Step 2: Convert to Grayscale
# -----------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# -----------------------------
# Step 3: Initialize EasyOCR
# -----------------------------
reader = easyocr.Reader(
    ['en'], 
    gpu=False, 
    verbose=False   # IMPORTANT: keeps terminal clean
)

# -----------------------------
# Step 4: Extract Text
# -----------------------------
results = reader.readtext(gray)

print("\n===== EXTRACTED TEXT =====\n")
for detection in results:
    text = detection[1]
    confidence = detection[2]
    print(f"{text}  (Confidence: {confidence:.2f})")

# -----------------------------
# Step 5: Draw Bounding Boxes
# -----------------------------
for detection in results:
    bbox = detection[0]
    text = detection[1]

    top_left = tuple(map(int, bbox[0]))
    bottom_right = tuple(map(int, bbox[2]))

    cv2.rectangle(image, top_left, bottom_right, (0, 200, 0), 10)
    cv2.putText(image,text,
        (top_left[0], top_left[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 0, 255), 3
    )

# -----------------------------
# Step 6: Display Output Image
# -----------------------------
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Detected Text with Bounding Boxes")
plt.show()