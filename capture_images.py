import cv2
import os

os.makedirs("dataset/images", exist_ok=True)
cap = cv2.VideoCapture(0)
img_count = 0

print("\n--- INSTRUCTIONS ---")
print("Press 's' to take a photo of yourself wearing the helmet/vest.")
print("Press 'q' to quit when done (aim for 20-30 photos).\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Training Data", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        img_path = f"dataset/images/gear_{img_count}.jpg"
        cv2.imwrite(img_path, frame)
        print(f"📸 Saved: {img_path}")
        img_count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Captured {img_count} images successfully!")