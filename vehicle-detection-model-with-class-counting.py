import cv2
from ultralytics import YOLO
import cvzone

# Load the YOLOv8 model
model = YOLO(r"runs\kaggle\working\runs\detect\train\weights\best.pt", task="detect")

# Open video capture
cap = cv2.VideoCapture(r"media-sample\v1.mp4")

# Define colors for different vehicle classes (BGR format)
class_colors = {
    'car': (0, 255, 0),
    'big bus': (255, 50, 0),
    'bus-l-': (255, 150, 0),
    'bus-s-': (255, 200, 0),
    'small bus': (255, 250, 0),
    'truck-xl-': (0, 0, 255),
    'big truck': (0, 0, 220),
    'truck-l-': (0, 0, 190),
    'mid truck': (0, 50, 180),
    'truck-m-': (0, 70, 170),
    'small truck': (20, 90, 160),
    'truck-s-': (40, 110, 150),
}

# Default color for undefined classes
DEFAULT_COLOR = (128, 128, 128)  # Gray

# Dictionary to store vehicle counts
vehicle_counts = {class_name: 0 for class_name in class_colors.keys()}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Reset counts for each frame
    vehicle_counts = {class_name: 0 for class_name in class_colors.keys()}

    # Run inference on the frame
    results = model(frame)
    
    # Process each detection
    for result in results[0].boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        
        # Get confidence and class
        confidence = float(result.conf[0])
        class_id = int(result.cls[0])
        class_name = model.names[class_id]
        
        if confidence > 0.4:  # Confidence threshold
            # Get color for the detected class
            color = class_colors.get(class_name.lower(), DEFAULT_COLOR)
            
            # Increment count for this class
            vehicle_counts[class_name.lower()] += 1
            
            # Draw box with class-specific color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label with matching color
            label = f'{class_name} {confidence*100:.2f}%'
            
            # Calculate optimal text position
            text_y = max(y1 - 10, 20)  # Ensure text doesn't go above frame
            
            # Create background for text that matches vehicle class color
            cvzone.putTextRect(frame, 
                             label, 
                             (x1, text_y), 
                             scale=0.8,
                             thickness=1,
                             colorR=color,
                             colorT=(255, 255, 255),
                             offset=5,
                             border=2)
    

    legend_y = 30
    for class_name, color in class_colors.items():
        count = vehicle_counts[class_name]
        cv2.putText(frame, 
                   f"{class_name}: {count}", 
                   (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, 
                   color, 
                   2)
        legend_y += 20
    
    
    cv2.imshow("Vehicle Detection with Counts", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
