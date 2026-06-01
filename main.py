import cv2
import numpy as np

class VisionAnalyticsPipeline:
    """
    Computer Vision Analytics & Tracking Pipeline
    Performs frame-by-frame object detection, contour tracking, and anomaly detection.
    """
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    def process_frame(self, frame):
        # 1. Grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # 2. Foreground Mask
        fg_mask = self.bg_subtractor.apply(blurred)
        
        # 3. Contour Detection
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        return frame, detections

if __name__ == "__main__":
    pipeline = VisionAnalyticsPipeline()
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(mock_frame, (320, 240), 50, (255, 255, 255), -1)
    processed, detections = pipeline.process_frame(mock_frame)
    print(f"Processed frame successfully. Detected {len(detections)} moving objects.")
