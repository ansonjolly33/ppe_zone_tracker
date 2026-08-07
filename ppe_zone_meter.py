import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. SETUP MODEL & VARIABLES
# ---------------------------------------------------------
# Load pretrained YOLO (or replace "yolov8n.pt" with your custom trained model path e.g. "best.pt")
model = YOLO("yolov8n.pt")

zone_points = []
drawing_complete = False

def draw_polygon_event(event, x, y, flags, param):
    """Mouse callback to allow clicking points to define the safe zone."""
    global zone_points, drawing_complete
    
    if event == cv2.EVENT_LBUTTONDOWN and not drawing_complete:
        zone_points.append([x, y])
        print(f"Point added: ({x}, {y})")
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing_complete = True
        print(f"✅ Safe Zone finalized with {len(zone_points)} points.")

# ---------------------------------------------------------
# 2. START CAMERA & MOUSE LISTENER
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)
cv2.namedWindow("PPE & Safe Zone Monitor")
cv2.setMouseCallback("PPE & Safe Zone Monitor", draw_polygon_event)

print("\n--- INSTRUCTIONS ---")
print("1. Left-click on the camera view to draw your Safe Zone polygon points.")
print("2. Right-click anywhere when done drawing to lock the zone.")
print("3. Press 'r' to reset the drawing anytime.")
print("4. Press 'q' to quit.")
print("--------------------\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Make a copy for UI annotations
    display_frame = frame.copy()

    # Draw current interactive polygon points
    if len(zone_points) > 0:
        pts = np.array(zone_points, np.int32).reshape((-1, 1, 2))
        
        # Green outline if finalized, Yellow if drawing in progress
        poly_color = (0, 255, 0) if drawing_complete else (0, 255, 255)
        cv2.polylines(display_frame, [pts], isClosed=drawing_complete, color=poly_color, thickness=2)

        # Draw circles on clicked nodes
        for pt in zone_points:
            cv2.circle(display_frame, tuple(pt), 5, (0, 165, 255), -1)

    # ---------------------------------------------------------
    # 3. RUN YOLO & CHECK ZONE INTERSECTION
    # ---------------------------------------------------------
    if drawing_complete and len(zone_points) >= 3:
        polygon_np = np.array(zone_points, np.int32)
        
        # Run inference
        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            
            # Class 0 = Person (standard COCO model)
            if cls_id == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Foot/Standing point (bottom center of bounding box)
                foot_x = int((x1 + x2) / 2)
                foot_y = y2

                # Point-in-polygon test: >0 inside, 0 on edge, <0 outside
                is_inside = cv2.pointPolygonTest(polygon_np, (foot_x, foot_y), False)

                if is_inside < 0:
                    # 🚨 OUTSIDE SAFE ZONE
                    color = (0, 0, 255) # Red
                    label = "ALERT: OUTSIDE SAFE ZONE!"
                    cv2.putText(display_frame, label, (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    # ✅ SAFE
                    color = (0, 255, 0) # Green

                # Draw bounding box and foot point
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(display_frame, (foot_x, foot_y), 6, (255, 255, 0), -1)

    elif not drawing_complete:
        cv2.putText(display_frame, "Click points on video to draw Safe Zone. Right-click to complete.",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("PPE & Safe Zone Monitor", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        zone_points = []
        drawing_complete = False
        print("Zone reset. Draw again.")

cap.release()
cv2.destroyAllWindows()