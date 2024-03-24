import cv2
from tracker import EuclideanDistTracker
import time

tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("4.mp4") 
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=120)

object_timestamps = {}
object_counts = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width, _ = frame.shape
    roi = frame[300:600, 350:1200]  # roi範圍
    
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])
            
    boxes_ids = tracker.update(detections)

    # 初始化畫面上物體的數量和時間
    object_counts_in_frame = len(boxes_ids)
    current_time = time.time()

    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        
        if id not in object_timestamps:
            object_timestamps[id] = current_time
            object_counts[id] = 1
        else:
            object_counts[id] += 1

        # 更新時間
        dwell_time = current_time - object_timestamps[id]

        #cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2) 
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(roi, f"Time: {dwell_time:.2f}s", (x, y + h -60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
        cv2.putText(roi, f"Count: {object_counts_in_frame}", (x, y + h + 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
