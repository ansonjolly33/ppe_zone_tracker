import cv2

def find_usb_webcam():
    print("Searching for connected USB webcam...")
    
    # Check indexes 0 through 4 with Media Foundation & DirectShow backends
    backends = [
        ("MSMF (Windows Media)", cv2.CAP_MSMF),
        ("DSHOW (DirectShow)", cv2.CAP_DSHOW)
    ]
    
    for index in range(5):
        for name, backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Found working USB Webcam at Index [{index}] using {name}")
                    return cap
                cap.release()
                
    return None

cap = find_usb_webcam()

if cap is None:
    print("\n❌ Could not connect to the USB Webcam.")
    print("Troubleshooting Steps:")
    print("1. Unplug and replug the USB webcam into a main USB port (back of PC).")
    print("2. Check Windows Privacy: Settings > Privacy & Security > Camera > Turn ON 'Let desktop apps access your camera'.")
    print("3. Ensure you aren't using Microsoft Store Python if restricted by Windows sandbox policy.")
else:
    print("Camera feed active! Press 'q' on the video window to close.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("USB Webcam Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()