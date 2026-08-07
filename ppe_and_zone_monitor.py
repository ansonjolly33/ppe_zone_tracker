import cv2
import numpy as np
import winsound
import time
import os
import threading
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. INITIALIZATION & SETUP
# ---------------------------------------------------------
if os.path.exists("best.pt"):
    model_path = "best.pt"
elif os.path.exists("ppe.pt"):
    model_path = "ppe.pt"
else:
    model_path = "yolov8n.pt"

model = YOLO(model_path)

print("\n---------------------------------------------------")
print(f"LOADED MODEL: {model_path}")
print("MODEL CLASSES:", model.names)
print("---------------------------------------------------\n")

zone_points = []
drawing_complete = False
last_alarm_time = 0

os.makedirs("violations_log", exist_ok=True)

def play_custom_sound(freq, duration):
    """Non-blocking sound trigger."""
    def _beep():
        try:
            winsound.Beep(int(freq), int(duration))
        except Exception as e:
            pass

    threading.Thread(target=_beep, daemon=True).start()

def mouse_click_handler(event, x, y, flags, param):
    """Mouse listener for zone creation."""
    global zone_points, drawing_complete
    if event == cv2.EVENT_LBUTTONDOWN and not drawing_complete:
        zone_points.append([x, y])
        print(f"Point added: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN and len(zone_points) >= 3:
        drawing_complete = True
        print(f"✅ Safe Zone locked with {len(zone_points)} points.")

cap = cv2.VideoCapture(0)
cv2.namedWindow("PPE & Safe Zone Integration")
cv2.setMouseCallback("PPE & Safe Zone Integration", mouse_click_handler)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    # Draw Safe Zone Boundary
    if len(zone_points) > 0:
        pts = np.array(zone_points, np.int32).reshape((-1, 1, 2))
        poly_color = (0, 255, 0) if drawing_complete else (0, 255, 255)
        cv2.polylines(display_frame, [pts], isClosed=drawing_complete, color=poly_color, thickness=2)
        for pt in zone_points:
            cv2.circle(display_frame, tuple(pt), 5, (0, 165, 255), -1)

    if drawing_complete and len(zone_points) >= 3:
        polygon_np = np.array(zone_points, dtype=np.int32).reshape((-1, 1, 2))
        
        # Create full frame zone mask
        h, w = frame.shape[:2]
        zone_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(zone_mask, [polygon_np], 255)
        
        results = model(frame, conf=0.25, verbose=False)[0]

        people_boxes = []
        ppe_boxes = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label_name = str(model.names[cls_id]).lower()
            coords = list(map(int, box.xyxy[0]))

            # FLEXIBLE CLASS MATCHING: Catches standard COCO 'person' or custom dataset labels
            if any(p in label_name for p in ["person", "worker", "man", "human", "body", "user", "0"]):
                people_boxes.append((coords, label_name))
            else:
                ppe_boxes.append((coords, label_name))

        # FALLBACK: If custom model ONLY detects helmets/vests and NO person class, 
        # use detected PPE bounding boxes to infer person position!
        if len(people_boxes) == 0 and len(ppe_boxes) > 0:
            # Group gear into a pseudo-person box
            all_x = [b[0][0] for b in ppe_boxes] + [b[0][2] for b in ppe_boxes]
            all_y = [b[0][1] for b in ppe_boxes] + [b[0][3] for b in ppe_boxes]
            pseudo_person = [min(all_x), min(all_y), max(all_x), max(all_y) + 150]
            people_boxes.append((pseudo_person, "inferred_person"))

        out_of_zone_detected = False
        inside_missing_ppe_detected = False

        for (px1, py1, px2, py2), p_label in people_boxes:
            # Clamp coordinates to frame bounds
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(w, px2), min(h, py2)

            # Center-Mass Centroid
            center_x = int((px1 + px2) / 2.0)
            center_y = int((py1 + py2) / 2.0)
            centroid_inside = (cv2.pointPolygonTest(polygon_np, (center_x, center_y), False) >= 0)

            # Mask Overlap Calculation
            person_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(person_mask, (px1, py1), (px2, py2), 255, -1)
            
            intersection = cv2.bitwise_and(zone_mask, person_mask)
            overlap_pixels = cv2.countNonZero(intersection)
            person_area = max(1, (px2 - px1) * (py2 - py1))
            overlap_ratio = overlap_pixels / float(person_area)

            # Zone Entry Rule: Either center point is inside OR box has >15% overlap
            is_inside_zone = centroid_inside or (overlap_ratio >= 0.15)

            has_helmet = False
            has_vest = False

            for (gx1, gy1, gx2, gy2), gname in ppe_boxes:
                # Check gear overlap with person bounding box
                if not (gx2 < px1 or gx1 > px2 or gy2 < py1 or gy1 > py2):
                    if any(h in gname for h in ["helmet", "hard-hat", "hardhat", "cap"]) and "no-" not in gname and "no" not in gname:
                        has_helmet = True
                    if any(v in gname for v in ["vest", "jacket"]) and "no-" not in gname and "no" not in gname:
                        has_vest = True

            # Determine Violation Status
            if not is_inside_zone:
                color = (0, 0, 255)  # Red
                status_text = f"OUT OF ZONE ({int(overlap_ratio*100)}%)"
                out_of_zone_detected = True

            elif not has_helmet or not has_vest:
                color = (0, 255, 255)  # Yellow
                inside_missing_ppe_detected = True

                if not has_helmet and not has_vest:
                    status_text = "INSIDE ZONE: NO HELMET & VEST!"
                elif not has_helmet:
                    status_text = "INSIDE ZONE: NO HELMET!"
                else:
                    status_text = "INSIDE ZONE: NO VEST!"

            else:
                color = (0, 255, 0)  # Green
                status_text = "SAFE: HELMET & VEST OK"

            # Render Person Box & Status
            cv2.rectangle(display_frame, (px1, py1), (px2, py2), color, 2)
            cv2.circle(display_frame, (center_x, center_y), 5, (255, 255, 255), -1)
            cv2.putText(display_frame, status_text, (px1, max(py1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Draw Detected Gear
        for (gx1, gy1, gx2, gy2), gname in ppe_boxes:
            if "no" in gname:
                gear_color = (0, 165, 255)  # Orange for missing gear class
            else:
                gear_color = (0, 255, 255)  # Yellow for valid gear

            cv2.rectangle(display_frame, (gx1, gy1), (gx2, gy2), gear_color, 1)
            cv2.putText(display_frame, gname, (gx1, max(gy1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, gear_color, 1)

        # Sound Rules
        current_time = time.time()
        if out_of_zone_detected:
            if current_time - last_alarm_time > 0.25:
                play_custom_sound(freq=2500, duration=150)
                last_alarm_time = current_time
        elif inside_missing_ppe_detected:
            if current_time - last_alarm_time > 0.8:
                play_custom_sound(freq=1000, duration=350)
                last_alarm_time = current_time

    elif not drawing_complete:
        cv2.putText(display_frame, "Left-click to draw Safe Zone. Right-click to lock zone.",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("PPE & Safe Zone Integration", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        zone_points = []
        drawing_complete = False
        print("Zone reset.")

cap.release()
cv2.destroyAllWindows()