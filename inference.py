import cv2
import time
import math
from ultralytics import YOLO

# 1. Load custom trained model weights
model = YOLO(r"runs\detect\train-8\weights\best.pt")

# 2. Select input source: Use 0 for webcam, or specify a video file path (e.g. "sample_video.mp4")
source = 0  
cap = cv2.VideoCapture(source)

if not cap.isOpened():
    # Fallback to index 1 if index 0 is unavailable
    cap = cv2.VideoCapture(1)

prev_workers = []

while cap.isOpened():
    start_time = time.time()
    success, frame = cap.read()
    if not success:
        print("End of stream or camera disconnected.")
        break

    # Run inference with ultra-low confidence threshold to catch distant gear
    results = model(frame, conf=0.05)[0]

    persons = []
    hardhats = []
    goggles = []
    vests = []
    
    no_hardhats = []
    no_goggles = []
    no_vests = []

    raw_detections = []

    # --- PARSE ALL CLASS LABELS ---
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id].lower().strip()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        raw_detections.append((label, x1, y1, x2, y2, conf))

        # Check variations of class names
        if "person" in label:
            persons.append((x1, y1, x2, y2))
        elif "no-hardhat" in label or "no_hardhat" in label or "no hardhat" in label or "no-helmet" in label:
            no_hardhats.append((x1, y1, x2, y2))
        elif "hardhat" in label or "helmet" in label:
            hardhats.append((x1, y1, x2, y2))
        elif "no-goggles" in label or "no_goggles" in label or "no goggles" in label or "no-glass" in label:
            no_goggles.append((x1, y1, x2, y2))
        elif "goggles" in label or "goggle" in label or "glasses" in label:
            goggles.append((x1, y1, x2, y2))
        elif "no-vest" in label or "no_vest" in label or "no vest" in label:
            no_vests.append((x1, y1, x2, y2))
        elif "vest" in label:
            vests.append((x1, y1, x2, y2))

    # --- VISUAL DEBUG: DRAW ALL RAW DETECTED ITEMS ---
    for label, x1, y1, x2, y2, conf in raw_detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)

    # --- WORKER ZONE DYNAMICS ---
    current_raw_workers = []

    if len(persons) > 0:
        for (px1, py1, px2, py2) in persons:
            current_raw_workers.append((px1, py1, px2, py2))
    else:
        # Group headgear and vests into virtual worker bounding zones
        all_gear = hardhats + no_hardhats + goggles + no_goggles + vests + no_vests
        head_items = hardhats + no_hardhats + goggles + no_goggles
        
        items_to_group = head_items if len(head_items) > 0 else all_gear
        used_items = set()

        for i, item in enumerate(items_to_group):
            if i in used_items:
                continue
            ix1, iy1, ix2, iy2 = item
            icx, icy = (ix1 + ix2) // 2, (iy1 + iy2) // 2

            width = max(100, (ix2 - ix1) * 3)
            height = max(200, (iy2 - iy1) * 6)
            px1, py1 = max(0, icx - width // 2), max(0, iy1 - 20)
            px2, py2 = min(frame.shape[1], icx + width // 2), min(frame.shape[0], iy1 + height)

            current_raw_workers.append((px1, py1, px2, py2))

            for j, other_item in enumerate(items_to_group):
                oix1, oiy1, oix2, oiy2 = other_item
                oicx, oicy = (oix1 + oix2) // 2, (oiy1 + oiy2) // 2
                dist = math.sqrt((icx - oicx)**2 + (icy - oicy)**2)
                if dist < 120:
                    used_items.add(j)

    # --- TEMPORAL SMOOTHING ---
    smoothed_workers = []
    for (rx1, ry1, rx2, ry2) in current_raw_workers:
        matched = False
        rcx, rcy = (rx1 + rx2) // 2, (ry1 + ry2) // 2
        for prev in prev_workers:
            px1, py1, px2, py2 = prev["box"]
            pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
            if math.sqrt((rcx - pcx)**2 + (rcy - pcy)**2) < 70:
                ax1 = int(px1 * 0.5 + rx1 * 0.5)
                ay1 = int(py1 * 0.5 + ry1 * 0.5)
                ax2 = int(px2 * 0.5 + rx2 * 0.5)
                ay2 = int(py2 * 0.5 + ry2 * 0.5)
                smoothed_workers.append({"id": prev["id"], "box": (ax1, ay1, ax2, ay2), "missed_frames": 0})
                matched = True
                break
        if not matched:
            new_id = len(smoothed_workers) + 1 if not prev_workers else max([w["id"] for w in prev_workers]) + 1
            smoothed_workers.append({"id": new_id, "box": (rx1, ry1, rx2, ry2), "missed_frames": 0})

    for prev in prev_workers:
        if prev["id"] not in [s["id"] for s in smoothed_workers] and prev["missed_frames"] < 5:
            prev["missed_frames"] += 1
            smoothed_workers.append(prev)

    prev_workers = smoothed_workers

    # --- COMPLIANCE & VIOLATION EVALUATION ---
    total_workers = len(smoothed_workers)
    violations_count = 0

    for worker in smoothed_workers:
        worker_id = worker["id"]
        px1, py1, px2, py2 = worker["box"]
        buf = 60  # generous bounding pixel search zone

        # Check positive gear detections
        has_helmet = any(hx1 >= px1 - buf and hx2 <= px2 + buf and hy1 >= py1 - buf and hy2 <= py2 for hx1, hy1, hx2, hy2 in hardhats)
        has_goggles = any(gx1 >= px1 - buf and gx2 <= px2 + buf and gy1 >= py1 - buf and gy2 <= py2 for gx1, gy1, gx2, gy2 in goggles)
        has_vest = any(vx1 >= px1 - buf and vx2 <= px2 + buf and vy1 >= py1 - buf and vy2 <= py2 for vx1, vy1, vx2, vy2 in vests)

        # Check negative gear detections
        has_no_helmet = any(nx1 >= px1 - buf and nx2 <= px2 + buf and ny1 >= py1 - buf and ny2 <= py2 for nx1, ny1, nx2, ny2 in no_hardhats)
        has_no_goggles = any(ng1 >= px1 - buf and ng2 <= px2 + buf and ngy1 >= py1 - buf and ngy2 <= py2 for ng1, ngy1, ng2, ngy2 in no_goggles)
        has_no_vest = any(nv1 >= px1 - buf and nv2 <= px2 + buf and nvy1 >= py1 - buf and nvy2 <= py2 for nv1, nvy1, nv2, nvy2 in no_vests)

        # Compliance logic
        helmet_ok = has_helmet and not has_no_helmet
        goggles_ok = has_goggles and not has_no_goggles
        vest_ok = has_vest and not has_no_vest

        # Explicit negative overrides
        if has_no_helmet: helmet_ok = False
        if has_no_goggles: goggles_ok = False
        if has_no_vest: vest_ok = False

        # Collect list of all missing equipment for the worker
        missing_violations = []
        if not helmet_ok:
            missing_violations.append("No-Helmet")
        if not goggles_ok:
            missing_violations.append("No-Goggles")
        if not vest_ok:
            missing_violations.append("No-Vest")

        is_safe = (len(missing_violations) == 0)

        if is_safe:
            color = (0, 255, 0)  # Green
            status_text = f"W#{worker_id}: SAFE"
        else:
            violations_count += 1
            color = (0, 0, 255)  # Red
            status_text = f"W#{worker_id}: {', '.join(missing_violations)}"

        # Draw worker tracking box
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)

        # Draw violation text banner
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.rectangle(frame, (px1, max(0, py1 - 25)), (px1 + text_size[0] + 12, max(22, py1)), color, -1)
        cv2.putText(frame, status_text, (px1 + 5, max(15, py1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # --- DASHBOARD PANEL ---
    cv2.rectangle(frame, (15, 15), (380, 95), (0, 0, 0), -1)
    cv2.putText(frame, f"INDIVIDUAL MONITORS: {total_workers}", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if violations_count > 0:
        cv2.putText(frame, f"ALERTS: {violations_count} VIOLATIONS DETECTED", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "FLOOR STATUS: ALL COMPLIANT", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Industrial Safety PPE & Violation Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()