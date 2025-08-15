import cv2
import os
from cvzone.HandTrackingModule import HandDetector

# Create a directory to save captured hand images
output_dir = 'data_2_0/9'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize hand detector (from cvzone) without drawing
detector = HandDetector(maxHands=1, detectionCon=0.7)  # detectionCon can be adjusted for confidence

# Start capturing from the webcam
cap = cv2.VideoCapture(0)

# Initialize the frame count
count = 0
padding = 30

while count < 50:
    ret, frame = cap.read()
    
    if ret:
        # Detect hands in the frame without drawing
        hands, _ = detector.findHands(frame, draw=False)  # draw=False to avoid adding landmarks/bounding box

        if hands:
            # Get the bounding box of the first hand
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            # Add padding around the hand
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(frame.shape[1], x + w + padding) - x_pad  # Ensure it doesn't go beyond frame width
            h_pad = min(frame.shape[0], y + h + padding) - y_pad  # Ensure it doesn't go beyond frame height

            # Crop the hand region with padding from the frame
            hand_img = frame[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]

            # Display the cropped hand region
            cv2.imshow('Hand', hand_img)

            # Save the cropped hand image without bounding box and dots
            img_name = os.path.join(output_dir, f'hand_image_{count+1}.jpg')
            cv2.imwrite(img_name, hand_img)
            print(f'Captured {img_name}')

            count += 1

        # Wait for 100 milliseconds before capturing the next image
        # Press 'q' to quit the capturing process early
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        print("Failed to capture image")
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

print("Finished capturing hand images.")
